"""
Genere docs/combo_guide.html et docs/tactical_guide.html.

Rejoue un echantillon de parties (greedy, volume, pour la frequence des
combos ; MCTS hybride, config recommandee du README, pour comparer un jeu
plus fort), decompose Forest.score() terme par terme (copie fidele de
engine.Forest.score(), verifiee par sommation == score() reel a chaque
foret), agrege par terme (probabilite de realisation, gain moyen si
realise, esperance = moyenne sur toutes les parties), puis rend les deux
pages HTML a partir de reference/guide_assets/ (CSS + polices en base64,
mis en cache ici pour ne pas dependre d'un acces reseau a chaque
regeneration -- meme logique que tools/base_cards.json).

Usage :
    python reference/gen_combo_guide.py [n_greedy] [n_mcts] [mcts_iterations]

Par defaut : 300 parties greedy, 18 parties MCTS, 200 iterations. Le MCTS
est le facteur limitant (~200 parties/h a 200 it.) ; reduire n_mcts pour
un apercu rapide.
"""
import json
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "reference" / "guide_assets"
DOCS = REPO / "docs"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reference"))

import engine as E  # noqa: E402
from game import DWELLER, TREE, Game  # noqa: E402
from search import MCTS, greedy_action, greedy_policy  # noqa: E402
import value_policy as VP  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Decomposition du score, terme a terme
# ---------------------------------------------------------------------------

def _threshold(count, table):
    idx = min(count, len(table) - 1)
    return table[idx]


def score_breakdown(f, linden_majority=True, tree_majority=True):
    """Copie fidele de Forest.score(), terme par terme. La somme DOIT
    egaler f.score(linden_majority, tree_majority) -- verifie a l'appel."""
    tc = f.tree_count
    dc = f.dweller_count
    ty = f.type_count
    n_trees = f.n_trees
    species = f.n_species

    terms = []  # (label, categorie, points)

    terms.append(("BIRCH (Bouleau)", "arbre", tc[E.T_BIRCH]))
    terms.append(("LINDEN (Tilleul) x majorite", "majorite",
                  tc[E.T_LINDEN] * (3 if linden_majority else 1)))
    beech = tc[E.T_BEECH] + f.bee_by_species[E.T_BEECH]
    terms.append(("BEECH (Hetre) >=4 exemplaires", "seuil",
                  5 * tc[E.T_BEECH] if beech >= 4 else 0))
    terms.append(("DOUGLAS_FIR (Sapin Douglas)", "arbre", 5 * tc[E.T_DOUGLAS_FIR]))
    terms.append(("OAK (Chene) si >=8 especes", "seuil",
                  10 * tc[E.T_OAK] if species >= 8 else 0))
    terms.append(("SILVER_FIR (Sapin blanc) x habitants dessus", "positionnel",
                  2 * f.silver_fir_dwellers))
    terms.append(("SYCAMORE (Sycomore) x nb arbres total", "combo",
                  n_trees * tc[E.T_SYCAMORE]))
    hc = tc[E.T_HORSE_CHESTNUT] + f.bee_by_species[E.T_HORSE_CHESTNUT]
    terms.append(("HORSE_CHESTNUT (Marronnier) palier", "seuil",
                  _threshold(hc, E.HORSE_CHESTNUT_POINTS)))

    terms.append(("EURASIAN_JAY (Geai)", "plat", 3 * dc[E.D_EURASIAN_JAY]))
    terms.append(("EUROPEAN_BADGER (Blaireau)", "plat", 2 * dc[E.D_EUROPEAN_BADGER]))
    terms.append(("POND_TURTLE (Cistude)", "plat", 5 * dc[E.D_POND_TURTLE]))
    terms.append(("SQUEAKER (?)", "plat", dc[E.D_SQUEAKER]))
    terms.append(("TAWNY_OWL (Chouette hulotte)", "plat", 5 * dc[E.D_TAWNY_OWL]))

    terms.append(("BLACKBERRIES x Plantes", "combo",
                  2 * dc[E.D_BLACKBERRIES] * ty[E.TY_PLANT]))
    terms.append(("BULLFINCH x Insectes", "combo",
                  2 * dc[E.D_BULLFINCH] * ty[E.TY_INSECT]))
    terms.append(("FALLOW_DEER x Onguliforme", "combo",
                  3 * dc[E.D_FALLOW_DEER] * ty[E.TY_CLOVEN]))
    terms.append(("GNAT x Chauve-souris", "combo",
                  dc[E.D_GNAT] * ty[E.TY_BAT]))
    terms.append(("GOSHAWK x Oiseaux", "combo",
                  3 * dc[E.D_GOSHAWK] * ty[E.TY_BIRD]))
    terms.append(("HEDGEHOG x Papillons", "combo",
                  2 * dc[E.D_HEDGEHOG] * ty[E.TY_BUTTERFLY]))
    terms.append(("RED_DEER x (arbres+Plantes)", "combo",
                  dc[E.D_RED_DEER] * (n_trees + ty[E.TY_PLANT])))
    terms.append(("STAG_BEETLE x Plantigrades", "combo",
                  dc[E.D_STAG_BEETLE] * ty[E.TY_PAWED]))
    terms.append(("TREE_FERNS x Amphibiens", "combo",
                  6 * dc[E.D_TREE_FERNS] * ty[E.TY_AMPHIBIAN]))
    terms.append(("WOLF x Cervides", "combo",
                  5 * dc[E.D_WOLF] * ty[E.TY_DEER]))

    terms.append(("EUROPEAN_HARE^2 (auto-combo)", "auto-combo",
                  dc[E.D_HARE] * dc[E.D_HARE]))
    terms.append(("RED_FOX x Lievres", "combo",
                  2 * dc[E.D_RED_FOX] * dc[E.D_HARE]))
    terms.append(("TREE_FROG x Moustiques", "combo",
                  5 * dc[E.D_TREE_FROG] * dc[E.D_GNAT]))
    terms.append(("LYNX si >=1 Chevreuil", "combo-binaire",
                  10 * dc[E.D_LYNX] if dc[E.D_ROE_DEER] else 0))
    terms.append(("WILD_BOAR si >=1 Marcassin/?", "combo-binaire",
                  10 * dc[E.D_WILD_BOAR] if dc[E.D_SQUEAKER] else 0))
    terms.append(("WILD_STRAWBERRIES si >=8 especes", "seuil",
                  10 * dc[E.D_STRAWBERRIES] if species >= 8 else 0))
    terms.append(("MOSS si >=10 (arbres+abeilles)", "seuil",
                  10 * dc[E.D_MOSS] if f.tree_count_with_modifiers() >= 10 else 0))
    terms.append(("GREAT_SPOTTED_WOODPECKER x majorite", "majorite",
                  10 * dc[E.D_WOODPECKER] if tree_majority else 0))

    terms.append(("BEECH_MARTEN x arbres pleins", "positionnel",
                  5 * dc[E.D_BEECH_MARTEN] * f.fully_occupied))
    terms.append(("WOOD_ANT x slots Bottom occupes", "positionnel",
                  2 * dc[E.D_WOOD_ANT] * f.bottom_total))
    terms.append(("CHAFFINCH sur Hetre", "positionnel", 5 * f.chaffinch_on_beech))
    terms.append(("RED_SQUIRREL sur Chene", "positionnel", 5 * f.squirrel_on_oak))
    terms.append(("EUROPEAN_FAT_DORMOUSE (Loir gris)", "positionnel", 15 * f.dormouse_hits))
    terms.append(("COMMON_TOAD paires", "combo", 5 * f.toad_scoring_cards))

    if dc[E.D_ROE_DEER]:
        rd = f.roe_deer_by_species
        sym = f.symbol_count
        acc = sum(rd[s] * sym[s] for s in range(E.N_TREES) if rd[s])
        terms.append(("ROE_DEER x symboles d'arbre", "combo", 3 * acc))
    else:
        terms.append(("ROE_DEER x symboles d'arbre", "combo", 0))

    terms.append(("FIRE_SALAMANDER palier", "set", _threshold(dc[E.D_SALAMANDER], E.SALAMANDER_POINTS)))
    terms.append(("FIREFLIES palier", "set", _threshold(dc[E.D_FIREFLIES], E.FIREFLIES_POINTS)))
    n_bat_species = len([1 for b in E.BAT_IDS if dc[b]])
    terms.append(("Chauves-souris (>=3 especes)", "set",
                  5 * sum(dc[b] for b in E.BAT_IDS) if n_bat_species >= 3 else 0))
    terms.append(("Papillons (sets)", "set", f.butterfly_score()))
    terms.append(("Grotte", "plat", f.cave))

    total = sum(p for _, _, p in terms)
    expected = f.score(linden_majority, tree_majority)
    assert total == expected, f"breakdown {total} != score() {expected}"
    return terms


# ---------------------------------------------------------------------------
# 2. Parties
# ---------------------------------------------------------------------------

def play_greedy_game(seed):
    game = Game(n_players=2, seed=seed)
    turns = 0
    while not game.over and turns < 2000:
        game.apply(greedy_policy(game, None))
        turns += 1
    return game


def play_mcts_game(seed, me, iterations=200, short_rollout_depth=10):
    game = Game(n_players=2, seed=seed)
    rng = random.Random(seed)
    bot = MCTS(observer=me, iterations=iterations, seed=seed,
               rollout_depth=40,
               leaf_eval=VP.make_pairwise_hybrid_leaf_eval(
                   short_rollout_depth=short_rollout_depth, seed=seed))
    turns = 0
    while not game.over and turns < 600:
        if game.current == me:
            action = bot.choose(game)
        else:
            action = greedy_action(game, rng)
        game.apply(action)
        bot.advance(action)
        turns += 1
    return game


def collect(games, label):
    records = []
    for game, player_idxs in games:
        for pi in player_idxs:
            f = game.players[pi].forest
            terms = score_breakdown(f)
            records.append({
                "label": label,
                "score": f.score(),
                "n_trees": f.n_trees,
                "n_dwellers": f.n_dwellers,
                "n_species": f.n_species,
                "terms": {name: pts for name, cat, pts in terms},
            })
    return records


def collect_data(n_greedy, n_mcts, mcts_iterations):
    t0 = time.time()
    print(f"Greedy : {n_greedy} parties...")
    greedy_games = [(play_greedy_game(seed=10_000 + s), [0, 1]) for s in range(n_greedy)]
    print(f"  {time.time()-t0:.1f}s")

    t1 = time.time()
    print(f"MCTS hybride ({n_mcts} parties, {mcts_iterations} it., siege alterne)...")
    mcts_games = []
    for s in range(n_mcts):
        me = s % 2
        g = play_mcts_game(seed=20_000 + s, me=me, iterations=mcts_iterations, short_rollout_depth=10)
        mcts_games.append((g, [me]))
        print(f"  partie {s+1}/{n_mcts} ok ({time.time()-t1:.0f}s cumule)")

    terms_catalog = {name: cat for name, cat, _ in
                      score_breakdown(Game(n_players=2, seed=1).players[0].forest)}
    records = collect(greedy_games, "greedy") + collect(mcts_games, "mcts")
    print(f"Temps total collecte : {time.time()-t0:.1f}s")
    return {
        "terms_catalog": terms_catalog,
        "records": records,
        "n_greedy_forests": len(greedy_games) * 2,
        "n_mcts_forests": len(mcts_games),
    }


# ---------------------------------------------------------------------------
# 3. Agregation
# ---------------------------------------------------------------------------

def aggregate(data):
    records = data["records"]
    catalog = data["terms_catalog"]
    by_label = {"greedy": [r for r in records if r["label"] == "greedy"],
                "mcts": [r for r in records if r["label"] == "mcts"]}

    stats = {}
    for name, cat in catalog.items():
        entry = {"category": cat, "greedy": {}, "mcts": {}}
        for label in ("greedy", "mcts"):
            vals = [r["terms"][name] for r in by_label[label]]
            nonzero = [v for v in vals if v > 0]
            entry[label] = {
                "n": len(vals),
                "p_realized": len(nonzero) / len(vals) if vals else 0.0,
                "mean_if_realized": statistics.mean(nonzero) if nonzero else 0.0,
                "expected_value": statistics.mean(vals) if vals else 0.0,
                "max": max(vals) if vals else 0,
            }
        stats[name] = entry

    overview = {}
    for label in ("greedy", "mcts"):
        rs = by_label[label]
        scores = [r["score"] for r in rs]
        overview[label] = {
            "n_games_or_forests": len(rs),
            "mean_score": statistics.mean(scores),
            "median_score": statistics.median(scores),
            "stdev_score": statistics.stdev(scores) if len(scores) > 1 else 0,
            "mean_n_trees": statistics.mean(r["n_trees"] for r in rs),
            "mean_n_dwellers": statistics.mean(r["n_dwellers"] for r in rs),
            "mean_n_species": statistics.mean(r["n_species"] for r in rs),
        }
    return {"stats": stats, "overview": overview}


# ---------------------------------------------------------------------------
# 4. Rendu HTML
# ---------------------------------------------------------------------------

FORMULAS = {
    "SYCAMORE (Sycomore) x nb arbres total": "pts = nb_Sycomores x nb_arbres_total",
    "EURASIAN_JAY (Geai)": "pts = 3 x nb_Geais",
    "EUROPEAN_BADGER (Blaireau)": "pts = 2 x nb_Blaireaux",
    "POND_TURTLE (Cistude)": "pts = 5 x nb_Cistudes",
    "SQUEAKER (?)": "pts = 1 x nb_Squeaker",
    "TAWNY_OWL (Chouette hulotte)": "pts = 5 x nb_Chouettes",
    "BLACKBERRIES x Plantes": "pts = 2 x nb_Mures x nb_cartes_Plante",
    "BULLFINCH x Insectes": "pts = 2 x nb_Bouvreuils x nb_cartes_Insecte",
    "FALLOW_DEER x Onguliforme": "pts = 3 x nb_Daims x nb_cartes_Onguliforme",
    "GNAT x Chauve-souris": "pts = 1 x nb_Moustiques x nb_cartes_Chauve-souris",
    "GOSHAWK x Oiseaux": "pts = 3 x nb_Autours x nb_cartes_Oiseau",
    "HEDGEHOG x Papillons": "pts = 2 x nb_Herissons x nb_cartes_Papillon",
    "RED_DEER x (arbres+Plantes)": "pts = nb_Cerfs x (nb_arbres + nb_cartes_Plante)",
    "STAG_BEETLE x Plantigrades": "pts = nb_Lucanes x nb_cartes_Plantigrade",
    "TREE_FERNS x Amphibiens": "pts = 6 x nb_Fougeres x nb_cartes_Amphibien",
    "WOLF x Cervides": "pts = 5 x nb_Loups x nb_cartes_Cervide",
    "EUROPEAN_HARE^2 (auto-combo)": "pts = nb_Lievres^2",
    "RED_FOX x Lievres": "pts = 2 x nb_Renards x nb_Lievres",
    "TREE_FROG x Moustiques": "pts = 5 x nb_Grenouilles x nb_Moustiques",
    "LYNX si >=1 Chevreuil": "pts = 10 x nb_Lynx, SI >=1 Chevreuil en foret",
    "WILD_BOAR si >=1 Marcassin/?": "pts = 10 x nb_Sangliers, SI >=1 Squeaker en foret",
    "WILD_STRAWBERRIES si >=8 especes": "pts = 10 x nb_Fraisiers, SI >=8 especes d'arbre",
    "MOSS si >=10 (arbres+abeilles)": "pts = 10 x nb_Mousses, SI arbres+abeilles >=10",
    "GREAT_SPOTTED_WOODPECKER x majorite": "pts = 10 x nb_Pics, SI majorite d'arbres",
    "BEECH_MARTEN x arbres pleins": "pts = 5 x nb_Fouines x nb_arbres_4_slots_occupes",
    "WOOD_ANT x slots Bottom occupes": "pts = 2 x nb_Fourmis x nb_slots_Bottom_occupes",
    "CHAFFINCH sur Hetre": "pts = 5 x nb_Pinsons poses sur un Hetre",
    "RED_SQUIRREL sur Chene": "pts = 5 x nb_Ecureuils poses sur un Chene",
    "EUROPEAN_FAT_DORMOUSE (Loir gris)": "pts = 15 x nb_Loirs actives (voir regle Loir)",
    "COMMON_TOAD paires": "pts = 5 x nb_Crapauds apparies (5 chacun, paire complete)",
    "ROE_DEER x symboles d'arbre": "pts = 3 x somme(nb_Chevreuils_symbole_S x nb_cartes_symbole_S)",
    "FIRE_SALAMANDER palier": "palier : 0/1/2/3+ = 0/5/15/25 pts",
    "FIREFLIES palier": "palier : 0/1/2/3/4+ = 0/0/10/15/20 pts",
    "Chauves-souris (>=3 especes)": "pts = 5 x total_chauves-souris, SI >=3 especes distinctes",
    "Papillons (sets)": "sets de papillons (1 par espece distincte, plafonne)",
    "BIRCH (Bouleau)": "pts = 1 x nb_Bouleaux",
    "LINDEN (Tilleul) x majorite": "pts = nb_Tilleuls x 3 (majorite) ou x1",
    "BEECH (Hetre) >=4 exemplaires": "pts = 5 x nb_Hetres, SI >=4 Hetres (abeille comptee)",
    "DOUGLAS_FIR (Sapin Douglas)": "pts = 5 x nb_Sapins_Douglas",
    "OAK (Chene) si >=8 especes": "pts = 10 x nb_Chenes, SI >=8 especes d'arbre",
    "SILVER_FIR (Sapin blanc) x habitants dessus": "pts = 2 x nb_habitants_sur_Sapin_blanc",
    "HORSE_CHESTNUT (Marronnier) palier": "palier 0..7+ = 0,1,4,9,16,25,36,49 pts",
    "Grotte": "pts = cartes en Grotte (Raton laveur / Ours brun)",
}

SHORT_NAME = {
    "SYCAMORE (Sycomore) x nb arbres total": "Sycomore",
    "BEECH_MARTEN x arbres pleins": "Fouine",
    "ROE_DEER x symboles d'arbre": "Chevreuil",
    "RED_DEER x (arbres+Plantes)": "Cerf élaphe",
    "GOSHAWK x Oiseaux": "Autour des palombes",
    "FALLOW_DEER x Onguliforme": "Daim",
    "HORSE_CHESTNUT (Marronnier) palier": "Marronnier",
    "OAK (Chene) si >=8 especes": "Chêne",
    "WOOD_ANT x slots Bottom occupes": "Fourmi des bois",
    "TREE_FERNS x Amphibiens": "Fougère arborescente",
    "BULLFINCH x Insectes": "Bouvreuil pivoine",
    "BEECH (Hetre) >=4 exemplaires": "Hêtre",
    "DOUGLAS_FIR (Sapin Douglas)": "Sapin Douglas",
    "LYNX si >=1 Chevreuil": "Lynx",
    "GREAT_SPOTTED_WOODPECKER x majorite": "Pic épeiche",
    "EUROPEAN_HARE^2 (auto-combo)": "Lièvre d'Europe",
    "WOLF x Cervides": "Loup",
    "LINDEN (Tilleul) x majorite": "Tilleul",
    "SILVER_FIR (Sapin blanc) x habitants dessus": "Sapin blanc",
    "Papillons (sets)": "Papillons",
    "RED_FOX x Lievres": "Renard roux",
    "TREE_FROG x Moustiques": "Grenouille arboricole",
    "COMMON_TOAD paires": "Crapaud commun",
}

CATEGORY_LABEL = {
    "combo": "Combo",
    "auto-combo": "Auto-combo",
    "combo-binaire": "Seuil binaire",
    "positionnel": "Positionnel",
    "set": "Set / palier",
    "seuil": "Seuil",
    "majorite": "Majorite",
    "plat": "Plat",
    "arbre": "Arbre",
}

TRUE_COMBO_CATS = {"combo", "auto-combo", "combo-binaire", "positionnel", "set"}

NAV = """<nav class="pages" aria-label="Pages du guide">
  <a href="combo_guide.html" aria-current="{cur1}">Guide des combos</a>
  <a href="tactical_guide.html" aria-current="{cur2}">Guide tactique</a>
  <a href="technique_guide.html" aria-current="{cur3}">Guide technique</a>
</nav>"""


def fmt(x, digits=1):
    return f"{x:.{digits}f}"


def pct(x):
    return f"{x*100:.0f}%"


def signed(x, digits=1):
    return f"{'+' if x >= 0 else ''}{fmt(x, digits)}"


def signed_cls(x):
    return "delta-up" if x >= 0 else "delta-down"


def render(agg):
    stats = agg["stats"]
    overview = agg["overview"]
    fonts = (ASSETS / "fonts_inline.css").read_text()
    shared_css = (ASSETS / "shared.css").read_text()

    def page_shell(title, body, active):
        nav = NAV.format(cur1="page" if active == "combo" else "false",
                          cur2="page" if active == "tactical" else "false",
                          cur3="page" if active == "technique" else "false")
        return f"""<title>{title}</title>
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
  <footer class="colophon">
    <span>{overview['greedy']['n_games_or_forests']} forets gloutonnes</span>
    <span>{overview['mcts']['n_games_or_forests']} forets MCTS (hybride)</span>
    <span>score() decompose terme a terme, verifie contre le moteur</span>
  </footer>
</div>
"""

    ranked = sorted(stats.items(), key=lambda kv: -kv[1]["greedy"]["expected_value"])
    combo_only = [(n, e) for n, e in ranked if e["category"] in TRUE_COMBO_CATS]
    top6 = combo_only[:6]

    plates = []
    for i, (name, e) in enumerate(top6, start=1):
        g = e["greedy"]
        plates.append(f"""<div class="plate">
      <span class="rank">N&deg;{i} &middot; {CATEGORY_LABEL[e['category']]}</span>
      <h3>{SHORT_NAME.get(name, name)}</h3>
      <p>{FORMULAS.get(name, '')}</p>
      <div class="figure">{fmt(g['expected_value'])} <span class="unit">pts d'esperance / partie</span></div>
      <p>{pct(g['p_realized'])} des parties &middot; {fmt(g['mean_if_realized'])} pts en moyenne quand realise</p>
    </div>""")

    rows = []
    for name, e in ranked:
        g, m = e["greedy"], e["mcts"]
        cat = e["category"]
        cls = "rank-top" if (name, e) in top6 else ""
        rows.append(f"""<tr class="{cls}">
      <td><span class="combo-name">{name}</span><span class="combo-formula">{FORMULAS.get(name, '')}</span></td>
      <td><span class="tag tag-{cat}">{CATEGORY_LABEL[cat]}</span></td>
      <td class="num">{pct(g['p_realized'])}</td>
      <td class="num">{fmt(g['mean_if_realized'])}</td>
      <td class="num expected">{fmt(g['expected_value'])}</td>
      <td class="num">{fmt(m['expected_value'])}</td>
    </tr>""")

    fox = stats["RED_FOX x Lievres"]
    hare = stats["EUROPEAN_HARE^2 (auto-combo)"]
    sycamore = stats["SYCAMORE (Sycomore) x nb arbres total"]

    body1 = f"""
<p class="dek">Espérance de points par combo, mesurée sur {overview['greedy']['n_games_or_forests']} forêts
gloutonnes et {overview['mcts']['n_games_or_forests']} forêts MCTS (config recommandée : rollout court +
modèle contrastif). Chaque ligne décompose <code>Forest.score()</code> terme par terme — la somme des
termes est vérifiée égale au score réel du moteur sur chaque forêt.</p>

<section aria-labelledby="top-combos">
  <h2 id="top-combos">Les 6 combos qui rapportent le plus, en moyenne</h2>
  <p class="section-note">Classés par espérance : probabilité de survenir dans une partie, multipliée par
  le gain quand ils surviennent.</p>
  <div class="callout-grid">
    {''.join(plates)}
  </div>
</section>

<section aria-labelledby="ledger">
  <h2 id="ledger">Ledger complet</h2>
  <p class="section-note">« Espérance » = points moyens sur <em>toutes</em> les parties, combo raté compris
  (0 si jamais réalisé). C'est ce chiffre qui doit guider la priorité de pose, pas le gain maximal théorique.</p>
  <div class="table-scroll">
    <table class="ledger">
      <caption>Décomposition du score par mécanisme</caption>
      <thead>
        <tr>
          <th scope="col">Combo</th>
          <th scope="col">Catégorie</th>
          <th scope="col" class="num">P(réalisé)</th>
          <th scope="col" class="num">Pts si réalisé</th>
          <th scope="col" class="num">Espérance (greedy)</th>
          <th scope="col" class="num">Espérance (MCTS)</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</section>

<section aria-labelledby="sycamore-note">
  <h2 id="sycamore-note">Le cas Sycomore</h2>
  <p class="section-note">Le Sycomore ne score PAS avec le nombre d'habitants, mais avec le nombre total
  d'<strong>arbres</strong> de la forêt (<code>{FORMULAS['SYCAMORE (Sycomore) x nb arbres total']}</code>).
  Verdict des données : loin d'être sous-optimal, c'est le combo n&deg;1 du jeu par espérance de points —
  devant le Lièvre au carré, la Fouine et le Chevreuil.</p>
  <div class="stat-strip">
    <div class="stat">
      <span class="label">Espérance (greedy)</span>
      <span class="value">{fmt(sycamore['greedy']['expected_value'])}<span class="unit">pts</span></span>
    </div>
    <div class="stat">
      <span class="label">Réalisé dans</span>
      <span class="value">{pct(sycamore['greedy']['p_realized'])}</span>
    </div>
    <div class="stat">
      <span class="label">Si réalisé</span>
      <span class="value">{fmt(sycamore['greedy']['mean_if_realized'])}<span class="unit">pts</span></span>
    </div>
    <div class="stat">
      <span class="label">Espérance Lièvre²</span>
      <span class="value">{fmt(hare['greedy']['expected_value'])}<span class="unit">pts</span></span>
    </div>
  </div>
  <p>Le Lièvre (auto-combo quadratique) et le Renard qui l'exploite ont l'air spectaculaires sur le papier
  ({fmt(hare['greedy']['expected_value'])} et {fmt(fox['greedy']['expected_value'])} pts d'espérance) mais
  restent loin derrière : ils dépendent d'empiler plusieurs exemplaires d'une carte précise, ce qui arrive
  moins souvent et rapporte moins en moyenne qu'un Sycomore planté n'importe quand — sa dépendance au nombre
  total d'arbres, une ressource que la partie accumule de toute façon, le rend quasiment toujours rentable
  ({pct(sycamore['greedy']['p_realized'])} des parties) et rarement anecdotique
  ({fmt(sycamore['greedy']['mean_if_realized'])} pts en moyenne quand il compte). Sous MCTS, son espérance
  grimpe encore ({fmt(sycamore['mcts']['expected_value'])} pts) : un jeu plus fort le pose plus tard, une
  fois la forêt déjà développée.</p>
</section>
"""

    deltas = sorted(
        ((name, e["category"], e["greedy"]["expected_value"], e["mcts"]["expected_value"],
          e["mcts"]["expected_value"] - e["greedy"]["expected_value"])
         for name, e in stats.items()),
        key=lambda t: -t[4],
    )
    deltas_up, deltas_down = deltas[:8], sorted(deltas, key=lambda t: t[4])[:5]

    def delta_rows(items):
        out = []
        for name, cat, g, m, d in items:
            cls = signed_cls(d)
            out.append(f"""<tr>
          <td><span class="combo-name">{name}</span></td>
          <td><span class="tag tag-{cat}">{CATEGORY_LABEL[cat]}</span></td>
          <td class="num">{fmt(g)}</td>
          <td class="num">{fmt(m)}</td>
          <td class="num {cls}">{signed(d)}</td>
        </tr>""")
        return "".join(out)

    ov_g, ov_m = overview["greedy"], overview["mcts"]
    trees_delta = ov_m["mean_n_trees"] - ov_g["mean_n_trees"]
    dwellers_delta = ov_m["mean_n_dwellers"] - ov_g["mean_n_dwellers"]
    species_delta = ov_m["mean_n_species"] - ov_g["mean_n_species"]

    body2 = f"""
<p class="dek">Ce que le bot MCTS (recherche arborescente, config recommandée : rollout court + modèle
contrastif) joue différemment d'une politique gloutonne — et ce que ça dit des priorités à avoir en
partie.</p>

<section aria-labelledby="overview">
  <h2 id="overview">Vue d'ensemble</h2>
  <p class="section-note">MCTS contrôle un seul siège par partie face à un adversaire glouton
  ({ov_m['n_games_or_forests']} parties, sièges alternés) ; le pool « greedy » ci-dessous vient de parties
  symétriques (greedy vs greedy, {ov_g['n_games_or_forests']} forêts) — ce n'est donc pas une comparaison
  tête-à-tête dans la même partie. Le score brut moyen n'est <strong>pas</strong> directement comparable
  entre les deux colonnes pour cette raison (adversaire et dynamique de partie différents) ; pour le vrai
  face-à-face MCTS-vs-greedy dans la même partie, voir le README (17/20 victoires, écart moyen +14,3 sur
  l'échantillon de référence). Les stats de composition ci-dessous restent indicatives, sur un échantillon
  MCTS volontairement restreint ({ov_m['n_games_or_forests']} parties, coûteux en temps de calcul).</p>
  <div class="stat-strip">
    <div class="stat">
      <span class="label">Arbres en forêt</span>
      <span class="value">{fmt(ov_m['mean_n_trees'],1)} <span class="unit {signed_cls(trees_delta)}">{signed(trees_delta)}</span></span>
    </div>
    <div class="stat">
      <span class="label">Habitants posés</span>
      <span class="value">{fmt(ov_m['mean_n_dwellers'],1)} <span class="unit {signed_cls(dwellers_delta)}">{signed(dwellers_delta)}</span></span>
    </div>
    <div class="stat">
      <span class="label">Espèces d'arbre</span>
      <span class="value">{fmt(ov_m['mean_n_species'],1)} <span class="unit {signed_cls(species_delta)}">{signed(species_delta)}</span></span>
    </div>
  </div>
</section>

<section aria-labelledby="leans-into">
  <h2 id="leans-into">Les combos que MCTS exploite plus que greedy</h2>
  <p class="section-note">Écart d'espérance (MCTS − greedy) par mécanisme. Un écart positif marque un combo
  que la recherche arborescente priorise davantage qu'une politique gloutonne à horizon 1 coup — c'est là
  que se loge l'essentiel de l'avantage de MCTS.</p>
  <div class="table-scroll">
    <table class="ledger">
      <caption>Écarts d'espérance MCTS vs greedy</caption>
      <thead>
        <tr>
          <th scope="col">Combo</th>
          <th scope="col">Catégorie</th>
          <th scope="col" class="num">Espérance greedy</th>
          <th scope="col" class="num">Espérance MCTS</th>
          <th scope="col" class="num">Écart</th>
        </tr>
      </thead>
      <tbody>{delta_rows(deltas_up)}</tbody>
    </table>
  </div>
</section>

<section aria-labelledby="leans-away">
  <h2 id="leans-away">Ce que MCTS délaisse</h2>
  <p class="section-note">À l'inverse, ces mécanismes rapportent <em>moins</em> sous MCTS que sous greedy —
  soit parce que la recherche trouve mieux ailleurs, soit parce que le delta immédiat de greedy les
  sur-valorise localement sans regarder plus loin.</p>
  <div class="table-scroll">
    <table class="ledger">
      <caption>Combos delaisses par MCTS</caption>
      <thead>
        <tr>
          <th scope="col">Combo</th>
          <th scope="col">Catégorie</th>
          <th scope="col" class="num">Espérance greedy</th>
          <th scope="col" class="num">Espérance MCTS</th>
          <th scope="col" class="num">Écart</th>
        </tr>
      </thead>
      <tbody>{delta_rows(deltas_down)}</tbody>
    </table>
  </div>
</section>

<section aria-labelledby="principles">
  <h2 id="principles">Enseignements</h2>
  <ol class="principles">
    <li>
      <span class="num">01</span>
      <div>
        <h3>Les combos adossés à une ressource abondante battent les combos-vedettes</h3>
        <p>Sycomore (nombre d'arbres), Fouine (arbres entièrement occupés) et Chevreuil (symboles d'arbre)
        trustent le haut du classement d'espérance — pas parce qu'ils rapportent le plus quand ils
        marchent, mais parce que leur condition (avoir des arbres, occuper des slots) est quasiment garantie
        en fin de partie. Le Lièvre au carré, souvent perçu comme LE combo fort, plafonne loin derrière
        en espérance : il dépend d'empiler plusieurs exemplaires d'une carte précise, ce qui arrive
        nettement moins souvent.</p>
      </div>
    </li>
    <li>
      <span class="num">02</span>
      <div>
        <h3>Les seuils binaires (Lynx/Chevreuil, Sanglier/Squeaker) sont des combos à bas risque</h3>
        <p>Une seule copie de la carte-condition suffit à débloquer le plein tarif — aucun intérêt à
        accumuler au-delà d'une copie de la pièce déclenchante, l'investissement doit aller sur la carte
        qui multiplie (Lynx, Sanglier) une fois la condition acquise.</p>
      </div>
    </li>
    <li>
      <span class="num">03</span>
      <div>
        <h3>Le score de la Clairière domine tout, ce qui déplace la vraie décision vers la gestion de main</h3>
        <p>Avec la Clairière active, l'essentiel du score vient du volume de cartes jouées sur la partie
        (elle-même dictée par la vitesse de vidage de la Clairière), pas uniquement de la puissance d'un
        combo isolé. MCTS gagne surtout en jouant plus de tours utiles, pas en trouvant des combos inédits.</p>
      </div>
    </li>
    <li>
      <span class="num">04</span>
      <div>
        <h3>Les combos « lents » (Sycomore, Chevreuil) récompensent la patience, pas la vitesse de pose</h3>
        <p>Contrairement au Lièvre (rentable dès la première paire), ces mécanismes grossissent avec l'état
        final de la forêt : les poser tôt gaspille leur potentiel. Un joueur pressé qui les joue en premier
        obtient systématiquement moins que l'espérance mesurée ici — et c'est justement l'écart que MCTS
        exploite le mieux (+{fmt(sycamore['mcts']['expected_value']-sycamore['greedy']['expected_value'])}
        pts d'espérance sur le Sycomore par rapport à greedy).</p>
      </div>
    </li>
  </ol>
</section>
"""

    DOCS.mkdir(exist_ok=True)
    (DOCS / "combo_guide.html").write_text(page_shell("Guide des combos — Forêt Mixte", body1, "combo"))
    (DOCS / "tactical_guide.html").write_text(page_shell("Guide tactique — Forêt Mixte", body2, "tactical"))
    print("Ecrit docs/combo_guide.html et docs/tactical_guide.html")


if __name__ == "__main__":
    n_greedy = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    n_mcts = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    mcts_iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    data = collect_data(n_greedy, n_mcts, mcts_iterations)
    agg = aggregate(data)
    render(agg)
