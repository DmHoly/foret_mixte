"""Pts/copie et pts/cout pour les cartes a seuil binaire (Lynx, Sanglier)
compares aux combos multiplicatifs (cervides, Fouine), sur des parties E
vs E reellement jouees.

Question de Mehdi (18/08) : Lynx et Sanglier partagent la meme structure
(+10 pts fixes si un declencheur faible -- Chevreuil pour Lynx, Marcassin
=SQUEAKER pour Sanglier -- est present, peu importe sa quantite). Pour N
cartes investies (cout de pose inclus), un combo multiplicatif est-il
superieur, ou est-ce un souci de comptage de notre cote ? Voir
reference/MODELS.md, "Lynx vs Sanglier" pour le resultat : le Lynx tient
la comparaison (tranche du Daim), le Sanglier est objectivement plus
faible (cout double pour le meme bonus binaire), pas un artefact de
mesure.

Usage :
    python reference/gen_lynx_boar_efficiency.py [n_games]
"""
import random, statistics, sys, time
from pathlib import Path
REPO = Path("/home/user/foret_mixte")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reference"))

import engine as E
import scoring_ref as R
from cards import POSITIONS, DWELLERS
from game import Game
import search as S
import value_policy as VP
from collections import defaultdict

COST = {d.name: d.cost for d in DWELLERS}


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


def play_one(seed, tiebreak):
    game = Game(n_players=2, seed=seed)
    turns = 0
    while not game.over and turns < 600:
        action = S.greedy_action(game, None, tiebreak=tiebreak)
        game.apply(action)
        turns += 1
    return game


CARDS_OF_INTEREST = ["LYNX", "WILD_BOAR", "ROE_DEER", "RED_DEER", "FALLOW_DEER",
                      "WOLF", "SQUEAKER", "BEECH_MARTEN"]

if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    tiebreak = VP.make_pairwise_gbm_tiebreak()

    t0 = time.time()
    counts = defaultdict(int)      # card -> total copies across all forets
    points = defaultdict(int)      # card -> total points scored BY that card type
    n_forests = 0
    lynx_when_present = []   # pts total du/des Lynx quand n_lynx>=1
    boar_when_present = []

    for gi in range(n_games):
        game = play_one(80000 + gi, tiebreak)
        for seat in (0, 1):
            forest = game.players[seat].forest
            ref = to_ref(forest)
            opp = to_ref(game.players[1 - seat].forest)
            n_forests += 1

            n_roe = R.count_card_names(ref, ["ROE_DEER"])
            n_squeaker = R.count_card_names(ref, ["SQUEAKER"])
            n_lynx = R.count_card_names(ref, ["LYNX"])
            n_boar = R.count_card_names(ref, ["WILD_BOAR"])

            counts["ROE_DEER"] += n_roe
            counts["SQUEAKER"] += n_squeaker
            counts["LYNX"] += n_lynx
            counts["WILD_BOAR"] += n_boar
            for name in ["RED_DEER", "FALLOW_DEER", "WOLF", "BEECH_MARTEN"]:
                counts[name] += R.count_card_names(ref, [name])

            lynx_pts = 10 * n_lynx if n_roe > 0 else 0
            boar_pts = 10 * n_boar if n_squeaker > 0 else 0
            points["LYNX"] += lynx_pts
            points["WILD_BOAR"] += boar_pts
            if n_lynx > 0:
                lynx_when_present.append(lynx_pts / n_lynx)
            if n_boar > 0:
                boar_when_present.append(boar_pts / n_boar)

            fully_occ = R._fully_occupied_trees(ref)
            points["ROE_DEER"] += 3 * sum(
                R.count_tree_symbols(ref, [ref.trees[ti]["name"]])
                for name, pos, ti in ref.dwellers() if name == "ROE_DEER")
            points["RED_DEER"] += R.count_card_types(ref, ["Tree", "Plant"]) * R.count_card_names(ref, ["RED_DEER"])
            points["FALLOW_DEER"] += 3 * R.count_card_types(ref, ["ClovenhoofedAnimal"]) * R.count_card_names(ref, ["FALLOW_DEER"])
            points["WOLF"] += 5 * R.count_card_types(ref, ["Deer"]) * R.count_card_names(ref, ["WOLF"])
            points["BEECH_MARTEN"] += 5 * fully_occ * R.count_card_names(ref, ["BEECH_MARTEN"])

        if (gi + 1) % 50 == 0:
            print(f"  {gi+1}/{n_games} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{n_games} parties, {n_forests} forets -- {time.time()-t0:.0f}s\n")
    print(f"{'carte':14s} {'cout':>5s} {'copies_tot':>11s} {'pts_tot':>9s} "
          f"{'pts/copie':>10s} {'pts/cout':>9s}")
    for name in CARDS_OF_INTEREST:
        c = counts[name]
        p = points[name]
        cost = COST[name]
        pts_per_copy = p / c if c else 0.0
        pts_per_cost = pts_per_copy / cost if cost else float('inf')
        print(f"{name:14s} {cost:5d} {c:11d} {p:9d} {pts_per_copy:10.2f} "
              f"{pts_per_cost if cost else float('nan'):9.2f}")

    print(f"\nLYNX : pts moyen PAR COPIE quand au moins 1 present (donc trigger deja paye) : "
          f"{statistics.mean(lynx_when_present):.2f} (n={len(lynx_when_present)})")
    print(f"WILD_BOAR : idem : {statistics.mean(boar_when_present):.2f} (n={len(boar_when_present)})")
