"""
Genere docs/technique_guide.html : tactiques concretes minees depuis des
parties MCTS (config recommandee) jouees coup par coup, pas juste la
composition finale des forets (voir gen_combo_guide.py pour ca).

Suit chaque carte physique individuellement (id() Python) depuis son
entree en main jusqu'a sa resolution (jouee ou defaussee en paiement),
pour repondre a des questions qu'un simple score final ne peut pas
trancher :
  - tempo d'ouverture : pioche ou pose, tour par tour
  - quelles especes sont posees quasi immediatement (peu de tours en main)
  - quelles especes finissent le plus souvent en monnaie de paiement
  - taille de main effectivement gardee en cours de partie
  - a quelle frequence le choix (fige) de Clairiere tombe sur une carte de
    combo fort

Usage :
    python reference/gen_technique_guide.py [n_games] [iterations]
"""
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "reference" / "guide_assets"
DOCS = REPO / "docs"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reference"))

import engine as E  # noqa: E402
from game import DWELLER, TREE, Game, card_min_cost  # noqa: E402
from search import MCTS  # noqa: E402
import value_policy as VP  # noqa: E402

DWELLER_NAME = E.DWELLER_NAME
TREE_NAME = E.TREE_NAME

# Dwellers combo-forts (voir docs/combo_guide.html) utilises pour mesurer
# a quelle frequence le choix fige de Clairiere (toujours la moins chere)
# tombe dessus par coincidence.
TOP_COMBO_DIDS = {
    E.DWELLER_ID["BEECH_MARTEN"], E.DWELLER_ID["ROE_DEER"],
    E.DWELLER_ID["RED_DEER"], E.DWELLER_ID["GOSHAWK"],
    E.DWELLER_ID["FALLOW_DEER"], E.DWELLER_ID["WOOD_ANT"],
}

MIN_SAMPLES = 6  # especes : nombre min de resolutions pour figurer au classement


# ---------------------------------------------------------------------------
# 1. Instrumentation d'une partie
# ---------------------------------------------------------------------------

class Tracker:
    def __init__(self):
        self.first_seen = {}
        self.card_repr = {}
        self.resolutions = []
        self.opening_actions = []  # (own_turn_idx <= 15, action_kind)
        self.hand_sizes = []
        self.clearing_takes = []   # bool : la carte prise etait un combo fort
        self.own_turn_count = defaultdict(int)

    def note_card(self, card, player, own_turn_idx):
        cid = id(card)
        if cid not in self.first_seen:
            self.first_seen[cid] = (player, own_turn_idx)
            self.card_repr[cid] = card

    def resolve_card(self, card, own_turn_idx, outcome, played_did=None):
        cid = id(card)
        if cid not in self.first_seen:
            return
        _player, seen_turn = self.first_seen.pop(cid)
        held = own_turn_idx - seen_turn
        kind, a, b = self.card_repr.pop(cid)
        if kind == TREE:
            self.resolutions.append({"label": ("tree", TREE_NAME[a]),
                                      "held": held, "outcome": outcome})
        else:
            dids = {a[0], b[0]}
            if outcome == "played" and played_did is not None:
                self.resolutions.append({"label": ("dweller", DWELLER_NAME[played_did]),
                                          "held": held, "outcome": "played"})
            else:
                for did in dids:
                    self.resolutions.append({"label": ("dweller", DWELLER_NAME[did]),
                                              "held": held, "outcome": outcome})


def _action_kind(action):
    if action[0] in ("tree", "dweller", "draw"):
        return action[0]
    return action[0]


def play_tracked(seed, iterations=150, short_rollout_depth=10, max_turns=600):
    game = Game(n_players=2, seed=seed)
    bots = {
        seat: MCTS(observer=seat, iterations=iterations, seed=seed * 31 + seat,
                   rollout_depth=40,
                   leaf_eval=VP.make_pairwise_hybrid_leaf_eval(
                       short_rollout_depth=short_rollout_depth, seed=seed * 31 + seat))
        for seat in (0, 1)
    }
    tr = Tracker()
    for seat in (0, 1):
        for card in game.players[seat].hand:
            tr.note_card(card, seat, 0)

    total_turns = 0
    while not game.over and total_turns < max_turns:
        player = game.current
        own_idx = tr.own_turn_count[player]
        hand_before = list(game.players[player].hand)
        clearing_before = list(game.clearing)

        action = bots[player].choose(game)
        kind = _action_kind(action)

        tr.hand_sizes.append(len(hand_before))
        if own_idx <= 15:
            tr.opening_actions.append((own_idx, kind))

        if kind == "draw" and clearing_before:
            idx = min(range(len(clearing_before)), key=lambda i: card_min_cost(clearing_before[i]))
            taken = clearing_before[idx]
            did_set = {taken[1][0], taken[2][0]} if taken[0] == DWELLER else set()
            tr.clearing_takes.append(bool(did_set & TOP_COMBO_DIDS))

        game.apply(action)
        for b in bots.values():
            b.advance(action)

        hand_after = game.players[player].hand
        ids_after = {id(c) for c in hand_after}
        clearing_after_ids = {id(c) for c in game.clearing}

        if kind == "tree":
            for c in hand_before:
                if c[0] == TREE and c[1] == action[1] and id(c) not in ids_after:
                    tr.resolve_card(c, own_idx, "played")
                    break
        elif kind == "dweller":
            did = action[1]
            for c in hand_before:
                if c[0] == DWELLER and (c[1][0] == did or c[2][0] == did) and id(c) not in ids_after:
                    tr.resolve_card(c, own_idx, "played", played_did=did)
                    break

        for c in hand_before:
            cid = id(c)
            if cid in ids_after:
                continue
            if cid in tr.card_repr and cid in clearing_after_ids:
                tr.resolve_card(c, own_idx, "discarded")

        for c in hand_after:
            tr.note_card(c, player, own_idx + 1)

        tr.own_turn_count[player] += 1
        total_turns += 1

    for seat in (0, 1):
        for c in game.players[seat].hand:
            tr.resolve_card(c, tr.own_turn_count[seat], "unresolved_end")

    return tr


def collect(n_games, iterations):
    all_resolutions, all_opening, all_hand_sizes, all_clearing_takes = [], [], [], []
    t0 = time.time()
    for s in range(n_games):
        tr = play_tracked(seed=80000 + s, iterations=iterations)
        all_resolutions.extend(tr.resolutions)
        all_opening.extend(tr.opening_actions)
        all_hand_sizes.extend(tr.hand_sizes)
        all_clearing_takes.extend(tr.clearing_takes)
        print(f"  partie {s+1}/{n_games} ({time.time()-t0:.0f}s cumule)")
    return {"resolutions": all_resolutions, "opening": all_opening,
            "hand_sizes": all_hand_sizes, "clearing_takes": all_clearing_takes,
            "n_games": n_games, "iterations": iterations}


# ---------------------------------------------------------------------------
# 2. Agregation
# ---------------------------------------------------------------------------

def aggregate(d):
    res, opening = d["resolutions"], d["opening"]
    hand_sizes, clearing_takes = d["hand_sizes"], d["clearing_takes"]

    opening_by_turn = defaultdict(Counter)
    for turn_idx, kind in opening:
        opening_by_turn[turn_idx][kind] += 1
    opening_stats = []
    for t in sorted(opening_by_turn):
        c = opening_by_turn[t]
        total = sum(c.values())
        opening_stats.append({"turn": t, "n": total,
                               "p_draw": c.get("draw", 0) / total,
                               "p_play": (c.get("tree", 0) + c.get("dweller", 0)) / total})

    by_species = defaultdict(lambda: {"played": [], "discarded": 0, "unresolved": 0})
    for r in res:
        kind, name = r["label"]
        v = by_species[(kind, name)]
        if r["outcome"] == "played":
            v["played"].append(r["held"])
        elif r["outcome"] == "discarded":
            v["discarded"] += 1
        else:
            v["unresolved"] += 1

    species_stats = []
    for (kind, name), v in by_species.items():
        n_total = len(v["played"]) + v["discarded"]
        if n_total < MIN_SAMPLES:
            continue
        species_stats.append({
            "kind": kind, "name": name,
            "n_played": len(v["played"]), "n_discarded": v["discarded"], "n_total": n_total,
            "mean_turns_held_if_played": statistics.mean(v["played"]) if v["played"] else None,
            "discard_rate": v["discarded"] / n_total,
        })

    hand_size_stats = {
        "mean": statistics.mean(hand_sizes), "median": statistics.median(hand_sizes),
        "p10": statistics.quantiles(hand_sizes, n=10)[0] if len(hand_sizes) >= 10 else min(hand_sizes),
        "p90": statistics.quantiles(hand_sizes, n=10)[-1] if len(hand_sizes) >= 10 else max(hand_sizes),
        "min": min(hand_sizes), "max": max(hand_sizes),
    }

    clearing_stats = {"n": len(clearing_takes),
                       "p_top_combo": (sum(clearing_takes) / len(clearing_takes)) if clearing_takes else 0}

    return {
        "opening": opening_stats,
        "species_by_immediacy": sorted(
            (s for s in species_stats if s["mean_turns_held_if_played"] is not None),
            key=lambda s: s["mean_turns_held_if_played"]),
        "species_by_discard": sorted(species_stats, key=lambda s: -s["discard_rate"]),
        "hand_size": hand_size_stats,
        "clearing": clearing_stats,
        "n_games": d["n_games"], "iterations": d["iterations"],
    }


# ---------------------------------------------------------------------------
# 3. Rendu HTML
# ---------------------------------------------------------------------------

CATEGORY_LABEL_KIND = {"tree": "Arbre", "dweller": "Habitant"}

NAV = """<nav class="pages" aria-label="Pages du guide">
  <a href="combo_guide.html" aria-current="{cur1}">Guide des combos</a>
  <a href="tactical_guide.html" aria-current="{cur2}">Guide tactique</a>
  <a href="technique_guide.html" aria-current="{cur3}">Guide technique</a>
</nav>"""


def fmt(x, digits=1):
    return f"{x:.{digits}f}"


def pct(x):
    return f"{x*100:.0f}%"


def render(agg):
    fonts = (ASSETS / "fonts_inline.css").read_text()
    shared_css = (ASSETS / "shared.css").read_text()
    nav = NAV.format(cur1="false", cur2="false", cur3="page")

    def bar(p):
        return (f'<div class="bar-track"><div class="bar-fill" '
                f'style="width:{max(2,round(p*100))}%"></div></div>')

    opening_rows = "".join(f"""<tr>
      <td class="num">{o['turn']+1}</td>
      <td class="num">{pct(o['p_draw'])}</td>
      <td class="num">{pct(o['p_play'])}</td>
      <td class="bar-cell">{bar(o['p_play'])}</td>
    </tr>""" for o in agg["opening"])

    immediacy = agg["species_by_immediacy"][:12]
    immediacy_rows = "".join(f"""<tr>
      <td><span class="combo-name">{DWELLER_NAME_FR(s)}</span>
          <span class="combo-formula">{CATEGORY_LABEL_KIND[s['kind']]}</span></td>
      <td class="num">{fmt(s['mean_turns_held_if_played'], 2)}</td>
      <td class="num">{pct(s['discard_rate'])}</td>
      <td class="num">{s['n_total']}</td>
    </tr>""" for s in immediacy)

    discard = agg["species_by_discard"][:12]
    discard_rows = "".join(f"""<tr>
      <td><span class="combo-name">{DWELLER_NAME_FR(s)}</span>
          <span class="combo-formula">{CATEGORY_LABEL_KIND[s['kind']]}</span></td>
      <td class="num">{pct(s['discard_rate'])}</td>
      <td class="num">{fmt(s['mean_turns_held_if_played'], 2) if s['mean_turns_held_if_played'] is not None else '—'}</td>
      <td class="num">{s['n_total']}</td>
    </tr>""" for s in discard)

    hs = agg["hand_size"]
    cl = agg["clearing"]

    body = f"""
<p class="dek">Ce que le bot MCTS (config recommandée) fait réellement, coup par coup, sur
{agg['n_games']} parties suivies carte par carte — pas seulement le score final (voir
<a href="combo_guide.html">le guide des combos</a> pour ça). Chaque carte physique est suivie
depuis son entrée en main jusqu'à sa résolution (jouée, ou défaussée comme monnaie de paiement).</p>

<section aria-labelledby="opening">
  <h2 id="opening">Tempo d'ouverture : piocher ou poser ?</h2>
  <p class="section-note">Proportion pioche / pose à chacun des 16 premiers tours du joueur (pas du
  jeu — les tours adverses ne comptent pas). Pas de phase de pioche dogmatique : le bot pose dès
  qu'une carte vaut le coup, y compris au tour 1.</p>
  <div class="table-scroll">
    <table class="ledger">
      <caption>Tempo d'ouverture par tour</caption>
      <thead><tr><th scope="col">Tour</th><th scope="col" class="num">Pioche</th>
        <th scope="col" class="num">Pose</th><th scope="col">Pose (visuel)</th></tr></thead>
      <tbody>{opening_rows}</tbody>
    </table>
  </div>
</section>

<section aria-labelledby="immediacy">
  <h2 id="immediacy">Cartes posées quasi immédiatement</h2>
  <p class="section-note">Nombre moyen de tours passés en main avant d'être jouées (uniquement
  quand elles finissent par l'être). Proche de 0 = posée dès que possible, aucune raison d'attendre.</p>
  <div class="table-scroll">
    <table class="ledger">
      <caption>Immédiateté de pose par espèce</caption>
      <thead><tr><th scope="col">Carte</th><th scope="col" class="num">Tours en main</th>
        <th scope="col" class="num">Taux de défausse</th><th scope="col" class="num">n</th></tr></thead>
      <tbody>{immediacy_rows}</tbody>
    </table>
  </div>
</section>

<section aria-labelledby="discard">
  <h2 id="discard">Cartes qui finissent en monnaie de paiement</h2>
  <p class="section-note">Fraction des fois où la carte est défaussée pour payer une autre carte
  plutôt que jouée elle-même. Un taux élevé sur un arbre coûteux (Marronnier, Chêne, Sapin Douglas,
  Sapin blanc) est cohérent : au-delà de la diversité d'espèces déjà en jeu, sa valeur de paiement
  dépasse sa valeur plantée.</p>
  <div class="table-scroll">
    <table class="ledger">
      <caption>Taux de défausse par espèce</caption>
      <thead><tr><th scope="col">Carte</th><th scope="col" class="num">Taux de défausse</th>
        <th scope="col" class="num">Tours en main si jouée</th><th scope="col" class="num">n</th></tr></thead>
      <tbody>{discard_rows}</tbody>
    </table>
  </div>
</section>

<section aria-labelledby="handsize">
  <h2 id="handsize">Taille de main gardée en jeu</h2>
  <p class="section-note">Pas de coussin de sécurité observé : la main descend régulièrement à 0-1
  carte avant la pioche suivante. Le bot ne thésaurise pas.</p>
  <div class="stat-strip">
    <div class="stat"><span class="label">Médiane</span><span class="value">{fmt(hs['median'],0)}</span></div>
    <div class="stat"><span class="label">Moyenne</span><span class="value">{fmt(hs['mean'],1)}</span></div>
    <div class="stat"><span class="label">10e percentile</span><span class="value">{fmt(hs['p10'],0)}</span></div>
    <div class="stat"><span class="label">90e percentile</span><span class="value">{fmt(hs['p90'],0)}</span></div>
  </div>
</section>

<section aria-labelledby="clearing">
  <h2 id="clearing">Choix en Clairière : pas (encore) une tactique apprise</h2>
  <p class="section-note">Rappel : la carte prise en Clairière suit une règle fixe (la moins chère
  disponible), jamais une décision de recherche — voir le guide tactique pour la tentative
  d'exposer ce choix à MCTS, abandonnée (résultat négatif mesuré). Cette règle fixe tombe par
  coïncidence sur une carte de combo fort (Fouine, Chevreuil, Cerf, Autour, Daim, Fourmi des bois)
  dans <strong>{pct(cl['p_top_combo'])}</strong> des pioches en Clairière ({cl['n']} pioches
  observées) — la plupart des prises restent de la carte bon marché ordinaire, pas une prise
  ciblée.</p>
</section>
"""

    footer = f"""  <footer class="colophon">
    <span>{agg['n_games']} parties MCTS suivies coup par coup</span>
    <span>{agg['iterations']} itérations</span>
    <span>chaque carte physique suivie individuellement (entrée en main → résolution)</span>
  </footer>"""

    title = "Guide technique — Forêt Mixte"
    html = f"""<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{fonts}
{shared_css}
</style>
<div class="wrap">
  <header class="masthead">
    <div class="eyebrow">Foret Mixte &middot; notes de simulation</div>
    <h1 class="title">{title}</h1>
    {nav}
  </header>
  {body}
{footer}
</div>
"""
    DOCS.mkdir(exist_ok=True)
    (DOCS / "technique_guide.html").write_text(html)
    print("Ecrit docs/technique_guide.html")


FR_NAMES = {
    "EUROPEAN_FAT_DORMOUSE": "Loir gris", "EUROPEAN_BADGER": "Blaireau européen",
    "LARGE_TORTOISESHELL": "Grande tortue", "VIOLET_CARPENTER_BEE": "Xylocope violet",
    "TREE_FERNS": "Fougère arborescente", "SQUEAKER": "Squeaker", "GNAT": "Moustique",
    "FIRE_SALAMANDER": "Salamandre tachetée", "BROWN_LONG_EARED_BAT": "Oreillard roux",
    "MOSS": "Mousse", "MOLE": "Taupe", "BROWN_BEAR": "Ours brun", "PENNY_BUN": "Cèpe de Bordeaux",
    "HORSE_CHESTNUT": "Marronnier", "OAK": "Chêne", "BECHSTEINS_BAT": "Murin de Bechstein",
    "DOUGLAS_FIR": "Sapin Douglas", "RED_FOX": "Renard roux", "STAG_BEETLE": "Lucane",
    "SILVER_FIR": "Sapin blanc", "RACCOON": "Raton laveur",
    "BARBASTELLE_BAT": "Barbastelle", "FLY_AGARIC": "Amanite tue-mouches",
    "GREATER_HORSESHOE_BAT": "Grand rhinolophe", "GREAT_SPOTTED_WOODPECKER": "Pic épeiche",
    "CHANTERELLE": "Girolle", "PARASOL_MUSHROOM": "Coulemelle", "WOLF": "Loup",
    "LYNX": "Lynx", "WILD_BOAR": "Sanglier", "COMMON_TOAD": "Crapaud commun",
    "TREE_FROG": "Grenouille arboricole", "WOOD_ANT": "Fourmi des bois",
    "BEECH_MARTEN": "Fouine", "ROE_DEER": "Chevreuil", "RED_DEER": "Cerf élaphe",
    "FALLOW_DEER": "Daim", "GOSHAWK": "Autour des palombes", "EUROPEAN_HARE": "Lièvre d'Europe",
    "CAMBERWELL_BEAUTY": "Morio", "PEACOCK_BUTTERFLY": "Paon-du-jour",
    "PURPLE_EMPEROR": "Grand Mars changeant", "SILVER_WASHED_FRITILLARY": "Tabac d'Espagne",
    "HEDGEHOG": "Hérisson", "RED_SQUIRREL": "Écureuil roux", "CHAFFINCH": "Pinson des arbres",
    "BULLFINCH": "Bouvreuil pivoine", "EURASIAN_JAY": "Geai des chênes",
    "TAWNY_OWL": "Chouette hulotte", "POND_TURTLE": "Cistude d'Europe",
    "WILD_STRAWBERRIES": "Fraisiers sauvages", "BLACKBERRIES": "Mûres", "FIREFLIES": "Lucioles",
    "BEECH": "Hêtre", "BIRCH": "Bouleau", "LINDEN": "Tilleul", "SYCAMORE": "Sycomore",
}


def DWELLER_NAME_FR(s):
    return FR_NAMES.get(s["name"], s["name"])


if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    print(f"Collecte : {n_games} parties MCTS ({iterations} it.)...")
    data = collect(n_games, iterations)
    agg = aggregate(data)
    render(agg)
