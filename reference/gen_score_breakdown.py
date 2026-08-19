"""Decompose le score final des forets de E (greedy + tiebreak GBM, le bot
le plus fort du depot) par mecanisme de scoring, sur des parties reellement
jouees en self-play.

Question de Mehdi (18/08) : quels sont les gros leviers de points -- point
brut additif, mecaniques multiplicatives (cervides, lapin/renard, etc.),
series a paliers (Marronnier, papillons...), seuils tout-ou-rien (Lynx,
chauve-souris...) ? Reutilise `scoring_ref.py` (l'oracle de scoring
litteral, une regle = une fonction, voir sa docstring) plutot que le
moteur rapide `engine.py` : c'est le seul des deux ou chaque regle est
assez isolee pour etre instrumentee sans dupliquer sa logique.

`score_forest_breakdown` est une copie carte-a-carte de
`scoring_ref.score_forest`, mais accumule les points par carte/mecanisme
au lieu de les sommer directement -- toute divergence entre les deux doit
etre corrigee ici en priorite si `scoring_ref.py` change.

Usage :
    python reference/gen_score_breakdown.py [n_games]
"""
import random, statistics, sys, time
from pathlib import Path
REPO = Path("/home/user/foret_mixte")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reference"))

import engine as E
import scoring_ref as R
from cards import POSITIONS
from game import Game
import search as S
import value_policy as VP
from collections import Counter, defaultdict


def to_ref(forest):
    ref = R.RefForest()
    for tid in forest.species:
        ref.add_tree(E.TREE_NAME[tid])
    for tree_idx in range(len(forest.species)):
        for pos_id, pos_name in enumerate(POSITIONS):
            for did, _symbol in forest.slots[tree_idx][pos_id]:
                ref.add_dweller(tree_idx, pos_name, E.DWELLER_NAME[did])
    ref.cave = forest.cave
    return ref


def score_forest_breakdown(forest, opponents=()):
    """Copie de R.score_forest, mais accumule les points dans un dict
    {source: points} au lieu d'un seul total."""
    all_forests = [forest, *opponents]
    bd = Counter()

    linden_count = R.count_card_names(forest, ["LINDEN"])
    linden_majority = linden_count >= max(
        R.count_card_names(f, ["LINDEN"]) for f in all_forests)
    tree_count_mod = R.count_card_types(forest, ["Tree"], ignore_modifiers=False)
    tree_majority = tree_count_mod >= max(
        R.count_card_types(f, ["Tree"], ignore_modifiers=False) for f in all_forests)

    species = R.count_tree_species(forest)
    beech_count = R.count_card_names(forest, ["BEECH"])

    for t in forest.trees:
        name = t["name"]
        if name == "BIRCH":
            bd["BIRCH"] += 1
        elif name == "LINDEN":
            bd["LINDEN"] += 3 if linden_majority else 1
        elif name == "BEECH":
            bd["BEECH"] += 5 if beech_count >= 4 else 0
        elif name == "DOUGLAS_FIR":
            bd["DOUGLAS_FIR"] += 5
        elif name == "OAK":
            bd["OAK"] += 10 if species >= 8 else 0
        elif name == "SILVER_FIR":
            bd["SILVER_FIR"] += 2 * sum(len(t["slots"][p]) for p in POSITIONS)
        elif name == "SYCAMORE":
            bd["SYCAMORE"] += R.count_card_types(forest, ["Tree"])

    hc = R.count_card_names(forest, ["HORSE_CHESTNUT"])
    if hc:
        bd["HORSE_CHESTNUT"] += R.score_by_count(hc, R.HORSE_CHESTNUT_POINTS)

    fully_occupied = R._fully_occupied_trees(forest)
    bottom_total = R._bottom_dwellers(forest)
    bat_points = R.score_bats(forest)
    seen_set_cards = set()

    for name, pos, ti in forest.dwellers():
        host = forest.trees[ti]["name"]
        if name in R.BAT_NAMES:
            bd["BATS"] += bat_points
        elif name in R.BUTTERFLY_NAMES:
            pass
        elif name == "BEECH_MARTEN":
            bd["BEECH_MARTEN"] += 5 * fully_occupied
        elif name == "BLACKBERRIES":
            bd["BLACKBERRIES"] += 2 * R.count_card_types(forest, ["Plant"])
        elif name == "BULLFINCH":
            bd["BULLFINCH"] += 2 * R.count_card_types(forest, ["Insect"])
        elif name == "CHAFFINCH":
            bd["CHAFFINCH"] += 5 if host == "BEECH" else 0
        elif name == "COMMON_TOAD":
            toads = forest.trees[ti]["slots"]["Bottom"].count("COMMON_TOAD")
            bd["COMMON_TOAD"] += 5 if toads > 1 else 0
        elif name == "EURASIAN_JAY":
            bd["EURASIAN_JAY"] += 3
        elif name == "EUROPEAN_BADGER":
            bd["EUROPEAN_BADGER"] += 2
        elif name == "EUROPEAN_FAT_DORMOUSE":
            opposite = "Right" if pos == "Left" else "Left"
            has_bat = any(d in R.BAT_NAMES for d in forest.trees[ti]["slots"][opposite])
            bd["EUROPEAN_FAT_DORMOUSE"] += 15 if has_bat else 0
        elif name == "EUROPEAN_HARE":
            bd["EUROPEAN_HARE"] += R.count_card_names(forest, ["EUROPEAN_HARE"])
        elif name == "FALLOW_DEER":
            bd["FALLOW_DEER"] += 3 * R.count_card_types(forest, ["ClovenhoofedAnimal"])
        elif name == "FIRE_SALAMANDER":
            if "FIRE_SALAMANDER" not in seen_set_cards:
                seen_set_cards.add("FIRE_SALAMANDER")
                n = R.count_card_names(forest, ["FIRE_SALAMANDER"])
                bd["FIRE_SALAMANDER"] += R.score_by_count(n, R.FIRE_SALAMANDER_POINTS)
        elif name == "FIREFLIES":
            if "FIREFLIES" not in seen_set_cards:
                seen_set_cards.add("FIREFLIES")
                n = R.count_card_names(forest, ["FIREFLIES"])
                bd["FIREFLIES"] += R.score_by_count(n, R.FIREFLIES_POINTS)
        elif name == "GNAT":
            bd["GNAT"] += R.count_card_types(forest, ["Bat"])
        elif name == "GOSHAWK":
            bd["GOSHAWK"] += 3 * R.count_card_types(forest, ["Bird"])
        elif name == "GREAT_SPOTTED_WOODPECKER":
            bd["GREAT_SPOTTED_WOODPECKER"] += 10 if tree_majority else 0
        elif name == "HEDGEHOG":
            bd["HEDGEHOG"] += 2 * R.count_card_types(forest, ["Butterfly"])
        elif name == "LYNX":
            bd["LYNX"] += 10 if R.count_card_names(forest, ["ROE_DEER"]) > 0 else 0
        elif name == "MOSS":
            bd["MOSS"] += 10 if tree_count_mod >= 10 else 0
        elif name == "POND_TURTLE":
            bd["POND_TURTLE"] += 5
        elif name == "RED_DEER":
            bd["RED_DEER"] += R.count_card_types(forest, ["Tree", "Plant"])
        elif name == "RED_FOX":
            bd["RED_FOX"] += 2 * R.count_card_names(forest, ["EUROPEAN_HARE"])
        elif name == "RED_SQUIRREL":
            bd["RED_SQUIRREL"] += 5 if host == "OAK" else 0
        elif name == "ROE_DEER":
            bd["ROE_DEER"] += 3 * R.count_tree_symbols(forest, [host])
        elif name == "SQUEAKER":
            bd["SQUEAKER"] += 1
        elif name == "STAG_BEETLE":
            bd["STAG_BEETLE"] += R.count_card_types(forest, ["PawedAnimal"])
        elif name == "TAWNY_OWL":
            bd["TAWNY_OWL"] += 5
        elif name == "TREE_FERNS":
            bd["TREE_FERNS"] += 6 * R.count_card_types(forest, ["Amphibian"])
        elif name == "TREE_FROG":
            bd["TREE_FROG"] += 5 * R.count_card_names(forest, ["GNAT"])
        elif name == "WILD_BOAR":
            bd["WILD_BOAR"] += 10 if R.count_card_names(forest, ["SQUEAKER"]) > 0 else 0
        elif name == "WILD_STRAWBERRIES":
            bd["WILD_STRAWBERRIES"] += 10 if species >= 8 else 0
        elif name == "WOLF":
            bd["WOLF"] += 5 * R.count_card_types(forest, ["Deer"])
        elif name == "WOOD_ANT":
            bd["WOOD_ANT"] += 2 * bottom_total

    bd["BUTTERFLY_SETS"] += R.score_butterflies_total(forest)
    bd["CAVE"] += forest.cave
    return bd


LEVERS = {
    "Cervides (multiplicatif)": (
        "ROE_DEER", "RED_DEER", "FALLOW_DEER", "WOLF"),
    'Multiplicatif "autres types"': (
        "TREE_FERNS", "WOOD_ANT", "GOSHAWK", "BULLFINCH", "BLACKBERRIES",
        "HEDGEHOG", "STAG_BEETLE", "GNAT", "TREE_FROG"),
    "Arbres multiplicatifs": ("SYCAMORE", "SILVER_FIR"),
    "Arbres a seuil": ("OAK", "BEECH", "LINDEN", "MOSS"),
    "Fouine seule (BEECH_MARTEN)": ("BEECH_MARTEN",),
    "Series a paliers": (
        "HORSE_CHESTNUT", "FIRE_SALAMANDER", "FIREFLIES", "BUTTERFLY_SETS",
        "BATS"),
    "Seuil binaire divers": (
        "LYNX", "WILD_BOAR", "WILD_STRAWBERRIES",
        "GREAT_SPOTTED_WOODPECKER", "COMMON_TOAD",
        "EUROPEAN_FAT_DORMOUSE", "CHAFFINCH", "RED_SQUIRREL"),
    "Lapin / renard": ("EUROPEAN_HARE", "RED_FOX"),
    "Additif fixe": (
        "DOUGLAS_FIR", "BIRCH", "TAWNY_OWL", "POND_TURTLE", "EURASIAN_JAY",
        "EUROPEAN_BADGER", "SQUEAKER", "CAVE"),
}


def play_one(seed, tiebreak):
    game = Game(n_players=2, seed=seed)
    turns = 0
    while not game.over and turns < 600:
        action = S.greedy_action(game, None, tiebreak=tiebreak)
        game.apply(action)
        turns += 1
    return game


if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    tiebreak = VP.make_pairwise_gbm_tiebreak()

    t0 = time.time()
    total_bd = Counter()
    total_score = 0.0
    n_forests = 0
    for gi in range(n_games):
        game = play_one(80000 + gi, tiebreak)
        refs = [to_ref(game.players[s].forest) for s in (0, 1)]
        for seat in (0, 1):
            bd = score_forest_breakdown(refs[seat], opponents=[refs[1 - seat]])
            total_bd.update(bd)
            total_score += sum(bd.values())
            n_forests += 1
        if (gi + 1) % 20 == 0:
            print(f"  {gi+1}/{n_games} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{n_games} parties, {n_forests} forets, score total moyen/foret : "
          f"{total_score/n_forests:.1f} -- {time.time()-t0:.0f}s\n")

    print(f"{'source':28s} {'total':>8s} {'moy/foret':>10s} {'% du score':>10s}")
    for name, pts in total_bd.most_common():
        print(f"{name:28s} {pts:8d} {pts/n_forests:10.2f} {pts/total_score*100:9.1f}%")

    print(f"\n-- regroupe par levier --")
    lever_totals = []
    for lever, cards in LEVERS.items():
        pts = sum(total_bd.get(c, 0) for c in cards)
        lever_totals.append((lever, pts))
    lever_totals.sort(key=lambda x: -x[1])
    for lever, pts in lever_totals:
        print(f"{lever:32s} {pts/n_forests:8.1f} pts/foret  {pts/total_score*100:5.1f}%")
