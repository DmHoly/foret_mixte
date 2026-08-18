"""Génère docs/decisive_guide.html : que fait E (greedy + heuristiques +
tiebreak Gradient Boosting) sur ses MEILLEURES parties -- distingue les
choix ÉVIDENTS (delta exact suffit, un candidat clairement en tête) des
choix SERRÉS où le tiebreak tranche, et RÉELLEMENT DÉCISIFS (le tiebreak
choisit une carte différente de ce qu'un delta exact seul aurait joué).
C'est précisément ce mécanisme qui fait gagner E 74-76/100 contre B (voir
reference/MODELS.md, "Comparateur pairwise dans greedy_action") -- ce
guide documente CE QUE ça change concrètement, pas seulement que ça
gagne.

Méthode : auto-jeu E vs E instrumenté (comme reference/run_combo_log.py,
instrumentation externe, ne modifie pas search.py). À chaque décision de
pose, on recalcule le pick "delta exact seul" (sans tiebreak) en plus du
pick réel d'E, pour savoir si le tiebreak était : inactif (un seul
candidat net en tête), actif mais sans effet (le meilleur delta exact
gagne aussi le départage), ou VRAIMENT décisif (départage sur un candidat
différent). On garde les `top_pct` parties (par score final) et on
n'agrège les tableaux détaillés QUE sur celles-ci -- l'utilisateur a
demandé les meilleures parties, pas la moyenne.

Usage :
    python reference/gen_decisive_guide.py [n_games] [top_pct]
"""
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

import engine as EN  # noqa: E402
import game as G  # noqa: E402
from cards import POSITIONS  # noqa: E402
from search import greedy_action  # noqa: E402
import value_policy as VP  # noqa: E402
from gen_technique_guide import FR_NAMES  # noqa: E402 -- reutilise la table de traduction

TIEBREAK_MARGIN = 3.0

POS_FR = {"Top": "Haut", "Bottom": "Bas", "Left": "Gauche", "Right": "Droite"}


def fr(name):
    return FR_NAMES.get(name, POS_FR.get(name, name))


def card_label(action, forest=None):
    """`forest` (optionnel) : etat AVANT le coup, pour distinguer deux
    poses de la MEME carte a des positions differentes (ex. le tiebreak
    departage souvent l'emplacement, pas seulement l'espece -- sans ca,
    deux actions differentes affichent le meme libelle et le tableau des
    corrections a l'air de dire "X -> X", trompeur)."""
    if action[0] == "tree":
        return fr(EN.TREE_NAME[action[1]])
    if action[0] in ("dweller", "free_dweller"):
        _, did, tree_idx, pos = action
        base = fr(EN.DWELLER_NAME[did])
        if forest is not None and tree_idx < len(forest.species):
            species = fr(EN.TREE_NAME[forest.species[tree_idx]])
            pos_name = fr(POSITIONS[pos])
            return f"{base} ({pos_name} de {species} #{tree_idx+1})"
        return base
    return action[0]


def _candidate_gains(game):
    """Reimplementation EXTERNE (instrumentation) du calcul de gain exact
    de greedy_action (tree_bonus=6 par defaut, comme la config reelle
    d'E) -- pour logguer les candidats sans dupliquer la decision."""
    actions = game.legal_actions()
    plays = [a for a in actions if a[0] not in ("draw", "skip_effect")]
    if not plays:
        return {}, []
    player = game.players[game.current]
    forest = player.forest
    base = forest.score()
    n_trees = forest.n_trees
    gains = {}
    for a in plays:
        if a[0] == "tree":
            gain = forest.delta_tree(a[1]) + 6 / (1 + n_trees)
        elif a[0] == "cave_discard":
            gain = a[1]
        else:
            _, did, tree_idx, pos = a
            forest.add_dweller(tree_idx, pos, did)
            gain = forest.score() - base
            forest.undo_dweller(tree_idx, pos, did)
        gains[a] = gain
    return gains, plays


def run_one(seed, tiebreak, records):
    g = G.Game(n_players=2, seed=seed)
    turns = 0
    while not g.over and turns < 500:
        cur = g.current
        forest_ref = g.players[cur].forest
        gains, plays = _candidate_gains(g)
        if plays:
            best_gain = max(gains.values())
            naive_pick = max(gains, key=lambda a: gains[a])
            near = [a for a in plays if gains[a] >= best_gain - TIEBREAK_MARGIN]

            action = greedy_action(g, None, tiebreak=tiebreak)
            score_before = g.scores()[cur]
            g.apply(action)
            score_after = g.scores()[cur]

            # `action` peut sortir de `plays` (ex. ("draw",) via l'urgence
            # Clairiere ou le repli main-vide) : ce n'est pas le tiebreak qui
            # a tranche entre des candidats proches dans ce cas, c'est une
            # autre heuristique qui a court-circuite tout le groupe -- pas
            # compte comme "decisif" au sens de ce guide.
            decisive = len(near) >= 2 and action in near and action != naive_pick
            records.append({
                "seed": seed, "turn": turns, "player": cur,
                "n_near": len(near),
                "decisive": decisive,
                "picked_card": card_label(action, forest_ref),
                "picked_species": card_label(action),
                "naive_card": card_label(naive_pick, forest_ref),
                "near_species": sorted({card_label(a) for a in near}),
                "picked_gain": gains.get(action, None),
                "naive_gain": best_gain,
                "score_delta": score_after - score_before,
            })
        else:
            action = greedy_action(g, None, tiebreak=tiebreak)
            g.apply(action)
        turns += 1
    return g.scores(), turns


def collect(n_games, seed0=200000):
    tiebreak = VP.make_pairwise_gbm_tiebreak()
    all_records = []
    performances = []  # (seed, player, final_score, n_turns)
    t0 = time.time()
    for gi in range(n_games):
        seed = seed0 + gi
        game_records = []
        scores, turns = run_one(seed, tiebreak, game_records)
        for r in game_records:
            r["final_score"] = scores[r["player"]]
        all_records.extend(game_records)
        for p in range(2):
            performances.append((seed, p, scores[p], turns))
        if (gi + 1) % 50 == 0:
            print(f"  {gi+1}/{n_games} parties, {time.time()-t0:.0f}s ecoulees", flush=True)
    return all_records, performances


CACHE_PATH = Path(__file__).resolve().parent / "decisive_guide_cache.json"


def save_cache(all_records, performances):
    import json
    CACHE_PATH.write_text(json.dumps({"records": all_records, "performances": performances}))


def load_cache():
    import json
    data = json.loads(CACHE_PATH.read_text())
    performances = [tuple(p) for p in data["performances"]]
    return data["records"], performances


def aggregate(all_records, performances, top_pct=10):
    performances_sorted = sorted(performances, key=lambda p: -p[2])
    n_top = max(1, round(len(performances_sorted) * top_pct / 100))
    top_set = {(s, p) for s, p, _, _ in performances_sorted[:n_top]}
    cutoff_score = performances_sorted[n_top - 1][2]

    all_scores = [p[2] for p in performances]

    top_records = [r for r in all_records if (r["seed"], r["player"]) in top_set]
    other_records = [r for r in all_records if (r["seed"], r["player"]) not in top_set]

    def decision_stats(records):
        n = len(records)
        n_near = sum(1 for r in records if r["n_near"] >= 2)
        n_decisive = sum(1 for r in records if r["decisive"])
        return {"n": n, "n_near": n_near, "n_decisive": n_decisive,
                "p_near": n_near / n if n else 0.0,
                "p_decisive_given_near": n_decisive / n_near if n_near else 0.0,
                "p_decisive": n_decisive / n if n else 0.0}

    top_stats = decision_stats(top_records)
    other_stats = decision_stats(other_records)

    # Overrides les plus frequents : naive_card -> picked_card, uniquement
    # parties du top, uniquement decisions reellement decisives.
    overrides = Counter()
    override_gain_loss = defaultdict(list)
    for r in top_records:
        if r["decisive"]:
            key = (r["naive_card"], r["picked_card"])
            overrides[key] += 1
            override_gain_loss[key].append(r["picked_gain"] - r["naive_gain"])

    override_rows = []
    for (naive, picked), n in overrides.most_common(20):
        losses = override_gain_loss[(naive, picked)]
        override_rows.append({
            "naive": naive, "picked": picked, "n": n,
            "mean_gain_diff": statistics.mean(losses),
        })

    # Choix "evidents" (n_near < 2) les plus frequents dans le top --
    # contraste avec les choix serres ci-dessus.
    obvious_cards = Counter(
        r["picked_species"] for r in top_records if r["n_near"] < 2)

    # Quelques exemples concrets pour illustrer (les diffs de gain les
    # plus negatives = le tiebreak sacrifie le plus de gain immediat --
    # les cas les plus contre-intuitifs, donc les plus instructifs).
    examples = sorted(
        (r for r in top_records if r["decisive"]),
        key=lambda r: (r["picked_gain"] - r["naive_gain"]))[:10]

    return {
        "n_performances": len(performances),
        "n_games": len(performances) // 2,
        "top_pct": top_pct,
        "n_top": n_top,
        "cutoff_score": cutoff_score,
        "score_mean": statistics.mean(all_scores),
        "score_median": statistics.median(all_scores),
        "score_p90": statistics.quantiles(all_scores, n=10)[8] if len(all_scores) >= 10 else max(all_scores),
        "score_max": max(all_scores),
        "top_stats": top_stats,
        "other_stats": other_stats,
        "override_rows": override_rows,
        "obvious_cards": obvious_cards.most_common(15),
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Rendu HTML
# ---------------------------------------------------------------------------

NAV = """<nav class="pages" aria-label="Pages du guide">
  <a href="combo_guide.html">Guide des combos</a>
  <a href="tactical_guide.html">Guide tactique</a>
  <a href="technique_guide.html">Guide technique</a>
  <a href="decisive_guide.html" aria-current="page">Guide des choix décisifs</a>
</nav>"""


def fmt(x, digits=1):
    return f"{x:.{digits}f}"


def pct(x):
    return f"{x*100:.0f}%"


def render(agg):
    fonts = (ASSETS / "fonts_inline.css").read_text()
    shared_css = (ASSETS / "shared.css").read_text()

    override_rows = "".join(f"""<tr>
      <td>{r['naive']}</td>
      <td>{r['picked']}</td>
      <td class="num">{r['n']}</td>
      <td class="num">{fmt(r['mean_gain_diff'], 2)}</td>
    </tr>""" for r in agg["override_rows"])

    obvious_rows = "".join(f"""<tr>
      <td>{card}</td>
      <td class="num">{n}</td>
    </tr>""" for card, n in agg["obvious_cards"])

    example_items = "".join(f"""<li>
      <strong>Partie {r['seed']}, tour {r['turn']}, joueur {r['player']}</strong> --
      espèces presque à égalité : {', '.join(r['near_species'])}.
      Le delta exact seul aurait joué <em>{r['naive_card']}</em>
      (+{fmt(r['naive_gain'],1)} pts immédiats) ; le tiebreak a préféré
      <em>{r['picked_card']}</em> (+{fmt(r['picked_gain'],1)} pts immédiats,
      soit {fmt(r['picked_gain']-r['naive_gain'],1)} de moins tout de suite --
      un pari sur la valeur différée). Cette partie a fini à {r['final_score']} pts.
    </li>""" for r in agg["examples"])

    ts, os_ = agg["top_stats"], agg["other_stats"]

    body = f"""
<p class="dek">Ce que E (greedy + heuristiques + comparateur Gradient Boosting, voir
<a href="../reference/MODELS.md">reference/MODELS.md</a>) joue réellement sur ses
{agg['n_top']} meilleures parties (top {agg['top_pct']}%, score final ≥ {agg['cutoff_score']:.0f} pts),
sur {agg['n_games']} parties d'auto-jeu ({agg['n_performances']} performances individuelles).
Distingue les choix ÉVIDENTS (un candidat net en tête, le delta exact suffit) des choix SERRÉS
où plusieurs candidats sont à moins de {TIEBREAK_MARGIN:.0f} pts les uns des autres, et parmi
ceux-là les choix réellement DÉCISIFS où le comparateur appris joue une carte DIFFÉRENTE de ce
qu'un simple calcul de gain immédiat aurait choisi -- c'est ce mécanisme précis qui fait gagner
E contre l'ancien bot (voir le tableau de gating dans MODELS.md).</p>

<section aria-labelledby="calib">
  <h2 id="calib">Calibration des scores</h2>
  <div class="stat-strip">
    <div class="stat"><span class="label">Moyenne</span><span class="value">{fmt(agg['score_mean'],0)}</span></div>
    <div class="stat"><span class="label">Médiane</span><span class="value">{fmt(agg['score_median'],0)}</span></div>
    <div class="stat"><span class="label">90e percentile</span><span class="value">{fmt(agg['score_p90'],0)}</span></div>
    <div class="stat"><span class="label">Maximum</span><span class="value">{fmt(agg['score_max'],0)}</span></div>
    <div class="stat"><span class="label">Seuil top {agg['top_pct']}%</span><span class="value">{fmt(agg['cutoff_score'],0)}</span></div>
  </div>
</section>

<section aria-labelledby="freq">
  <h2 id="freq">À quelle fréquence le tiebreak intervient-il ?</h2>
  <p class="section-note">"Serré" = au moins deux candidats à moins de {TIEBREAK_MARGIN:.0f} pts
  l'un de l'autre en gain exact. "Décisif" = le tiebreak a effectivement choisi une carte
  différente de celle qu'un delta exact seul aurait jouée -- pas juste confirmé le même choix.</p>
  <div class="table-scroll">
    <table class="ledger">
      <caption>Fréquence des décisions serrées / décisives</caption>
      <thead><tr><th scope="col"></th><th scope="col" class="num">Décisions</th>
        <th scope="col" class="num">% serrées</th><th scope="col" class="num">% décisives</th>
        <th scope="col" class="num">% décisives (parmi serrées)</th></tr></thead>
      <tbody>
        <tr><td>Top {agg['top_pct']}% parties</td><td class="num">{ts['n']}</td>
          <td class="num">{pct(ts['p_near'])}</td><td class="num">{pct(ts['p_decisive'])}</td>
          <td class="num">{pct(ts['p_decisive_given_near'])}</td></tr>
        <tr><td>Reste des parties</td><td class="num">{os_['n']}</td>
          <td class="num">{pct(os_['p_near'])}</td><td class="num">{pct(os_['p_decisive'])}</td>
          <td class="num">{pct(os_['p_decisive_given_near'])}</td></tr>
      </tbody>
    </table>
  </div>
</section>

<section aria-labelledby="overrides">
  <h2 id="overrides">Ce que le tiebreak préfère, contre l'intuition du delta exact</h2>
  <p class="section-note">Paires (carte que le delta exact aurait jouée -> carte réellement jouée)
  les plus fréquentes parmi les décisions décisives des meilleures parties, avec la perte moyenne
  de gain immédiat encaissée pour ce pari (négatif = le tiebreak accepte moins de points tout de
  suite, en pariant sur une meilleure valeur différée).</p>
  <div class="table-scroll">
    <table class="ledger">
      <caption>Corrections les plus fréquentes du tiebreak</caption>
      <thead><tr><th scope="col">Delta exact aurait joué</th><th scope="col">Tiebreak joue</th>
        <th scope="col" class="num">n</th><th scope="col" class="num">Δ gain immédiat</th></tr></thead>
      <tbody>{override_rows}</tbody>
    </table>
  </div>
</section>

<section aria-labelledby="obvious">
  <h2 id="obvious">Les choix évidents : cartes jouées sans hésitation</h2>
  <p class="section-note">Cartes les plus souvent jouées quand un seul candidat domine nettement
  (aucun départage nécessaire) dans les meilleures parties -- le socle du jeu, pas la source du
  gain contre l'ancien bot.</p>
  <div class="table-scroll">
    <table class="ledger">
      <caption>Cartes jouées en situation évidente</caption>
      <thead><tr><th scope="col">Carte</th><th scope="col" class="num">n</th></tr></thead>
      <tbody>{obvious_rows}</tbody>
    </table>
  </div>
</section>

<section aria-labelledby="examples">
  <h2 id="examples">Exemples concrets : les paris les plus contre-intuitifs</h2>
  <p class="section-note">Les décisions décisives où le tiebreak a sacrifié le plus de points
  immédiats pour jouer une autre carte -- les cas les plus instructifs pour comprendre ce qu'il
  "voit" que le calcul à un coup ne voit pas.</p>
  <ul class="example-list">{example_items}</ul>
</section>
"""

    footer = f"""  <footer class="colophon">
    <span>{agg['n_games']} parties E vs E</span>
    <span>{agg['n_performances']} performances (2 par partie)</span>
    <span>top {agg['top_pct']}% par score final ({agg['n_top']} performances analysées en détail)</span>
  </footer>"""

    title = "Guide des choix décisifs — Forêt Mixte"
    html = f"""<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{fonts}
{shared_css}
.example-list {{ display: flex; flex-direction: column; gap: 0.75rem; padding-left: 1.25rem; }}
.example-list li {{ line-height: 1.5; }}
</style>
<div class="wrap">
  <header class="masthead">
    <div class="eyebrow">Foret Mixte &middot; notes de simulation</div>
    <h1 class="title">{title}</h1>
    {NAV}
  </header>
  {body}
{footer}
</div>
"""
    DOCS.mkdir(exist_ok=True)
    (DOCS / "decisive_guide.html").write_text(html)
    print("Ecrit docs/decisive_guide.html")


if __name__ == "__main__":
    arg1 = sys.argv[1] if len(sys.argv) > 1 else "500"
    top_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 10
    if arg1 == "--from-cache":
        print(f"Rendu depuis le cache ({CACHE_PATH})...")
        all_records, performances = load_cache()
    else:
        n_games = int(arg1)
        print(f"Collecte : {n_games} parties E vs E...")
        all_records, performances = collect(n_games)
        save_cache(all_records, performances)
        print(f"Cache sauvegarde dans {CACHE_PATH} (relancer avec --from-cache pour re-rendre sans resimuler)")
    agg = aggregate(all_records, performances, top_pct=top_pct)
    render(agg)
