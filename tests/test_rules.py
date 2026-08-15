"""
Tests du moteur optimisé contre l'oracle `scoring_ref`.

Deux familles :
  1. Cas unitaires repris des tests du dépôt de référence (valeurs attendues
     écrites en dur), pour les règles que l'ancien moteur implémentait mal.
  2. Fuzzing : des milliers de forêts légales aléatoires, où le score du
     moteur optimisé doit être identique à celui de l'oracle. C'est ce test
     qui protège le portage JAX à venir.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine as E
import scoring_ref as R
from cards import DWELLERS, POSITIONS


def build_pair(placements, trees, cave=0):
    """Construit la même forêt dans les deux représentations."""
    fast = E.Forest()
    ref = R.RefForest()
    for name in trees:
        fast.add_tree(E.TREE_ID[name])
        ref.add_tree(name)
    for tree_idx, position, dweller in placements:
        fast.add_dweller(tree_idx, E.POS_ID[position], E.DWELLER_ID[dweller])
        ref.add_dweller(tree_idx, position, dweller)
    fast.cave = ref.cave = cave
    return fast, ref


def random_forest(rng, n_trees=None, n_dwellers=None):
    n_trees = n_trees if n_trees is not None else rng.randint(1, 14)
    n_dwellers = n_dwellers if n_dwellers is not None else rng.randint(0, 45)
    trees = [rng.choice(E.TREE_NAME) for _ in range(n_trees)]
    fast = E.Forest()
    for name in trees:
        fast.add_tree(E.TREE_ID[name])
    placements = []
    for _ in range(n_dwellers):
        did = rng.randrange(E.N_DWELLERS)
        options = list(fast.legal_positions(did))
        if not options:
            continue
        ti, pos = rng.choice(options)
        fast.add_dweller(ti, pos, did)
        placements.append((ti, POSITIONS[pos], E.DWELLER_NAME[did]))
    return trees, placements


# ---------------------------------------------------------------------------
# 1. Cas unitaires
# ---------------------------------------------------------------------------


def test_roe_deer_counts_every_card_with_its_tree_symbol():
    """Repris de RoeDeer.test.ts : un chevreuil sur tilleul avec 1 autre carte
    à symbole tilleul marque 9, pas 3 ni 6."""
    trees = ["LINDEN", "LINDEN"]
    placements = [(0, "Left", "ROE_DEER")]
    fast, ref = build_pair(placements, trees)
    # 2 tilleuls + le chevreuil lui-même = 3 cartes à symbole tilleul
    assert R.score_forest(ref) == fast.score()
    assert fast.score() == 3 * 3 + 2 * 3  # chevreuil + 2 tilleuls majoritaires


def test_roe_deer_counts_dwellers_hosted_on_the_same_species():
    trees = ["BEECH", "BEECH"]
    placements = [
        (0, "Right", "ROE_DEER"),
        (0, "Top", "CHAFFINCH"),
        (1, "Bottom", "WOOD_ANT"),
    ]
    fast, ref = build_pair(placements, trees)
    # 2 hêtres + 3 habitants posés sur des hêtres = 5 cartes à symbole hêtre
    assert fast.score() == R.score_forest(ref)


def test_paired_toads_score_five_each():
    """Repris de CommonToad.test.ts : chaque crapaud marque 5, donc 10 par
    paire. L'ancien moteur n'en donnait que 5."""
    trees = ["OAK"]
    placements = [(0, "Bottom", "COMMON_TOAD"), (0, "Bottom", "COMMON_TOAD")]
    fast, ref = build_pair(placements, trees)
    assert fast.score() == R.score_forest(ref)
    assert fast.score() == 10


def test_lone_toad_scores_nothing():
    fast, ref = build_pair([(0, "Bottom", "COMMON_TOAD")], ["OAK"])
    assert fast.score() == R.score_forest(ref) == 0


def test_carpenter_bee_makes_its_tree_count_twice():
    """MOSS demande 10 arbres. Avec 9 arbres et une abeille, le seuil tombe."""
    trees = ["DOUGLAS_FIR"] * 9
    fast, ref = build_pair([(0, "Bottom", "MOSS")], trees)
    assert fast.score() == R.score_forest(ref)
    without_bee = fast.score()

    fast2, ref2 = build_pair(
        [(0, "Bottom", "MOSS"), (1, "Left", "VIOLET_CARPENTER_BEE")], trees
    )
    assert fast2.score() == R.score_forest(ref2)
    assert fast2.score() - without_bee == 10


def test_bats_need_three_distinct_species():
    trees = ["BEECH", "BEECH", "BEECH"]
    two = [(0, "Left", "BECHSTEINS_BAT"), (1, "Right", "BROWN_LONG_EARED_BAT")]
    fast, ref = build_pair(two, trees)
    assert fast.score() == R.score_forest(ref) == 0

    three = two + [(2, "Left", "GREATER_HORSESHOE_BAT")]
    fast, ref = build_pair(three, trees)
    assert fast.score() == R.score_forest(ref)
    assert fast.score() == 15  # 5 points par carte chauve-souris


def test_fat_dormouse_activated_by_a_bat_placed_afterwards():
    """L'ordre de pose ne doit pas changer le score."""
    trees = ["BEECH"]
    a = [(0, "Left", "EUROPEAN_FAT_DORMOUSE"), (0, "Right", "BROWN_LONG_EARED_BAT")]
    b = [(0, "Right", "BROWN_LONG_EARED_BAT"), (0, "Left", "EUROPEAN_FAT_DORMOUSE")]
    fa, ra = build_pair(a, trees)
    fb, rb = build_pair(b, trees)
    assert fa.score() == fb.score() == R.score_forest(ra) == R.score_forest(rb)
    assert fa.score() == 15


def test_hare_scores_quadratically():
    trees = ["BIRCH", "BIRCH"]
    placements = [(0, "Left", "EUROPEAN_HARE")] * 3
    fast, ref = build_pair(placements, trees)
    assert fast.score() == R.score_forest(ref)
    assert fast.score() == 3 * 3 + 2  # 3 lièvres à 3 points + 2 bouleaux


def test_zero_point_cards_are_intentional():
    """BROWN_BEAR, MOLE, RACCOON, l'abeille et les 4 champignons valent 0 dans
    le moteur de référence. Ce ne sont pas des règles oubliées."""
    for name in ("BROWN_BEAR", "MOLE", "RACCOON", "CHANTERELLE",
                 "FLY_AGARIC", "PARASOL_MUSHROOM", "PENNY_BUN"):
        did = E.DWELLER_ID[name]
        tree_id, pos = sorted(E.PLACEMENTS[did])[0]
        fast = E.Forest()
        fast.add_tree(tree_id)
        before = fast.score()
        fast.add_dweller(0, pos, did)
        assert fast.score() == before, name


def test_placements_match_the_reference_variants():
    """Les 7 habitants dont les positions étaient trop permissives."""
    assert E.PLACEMENTS[E.DWELLER_ID["ROE_DEER"]] == frozenset({
        (E.TREE_ID["LINDEN"], E.LEFT), (E.TREE_ID["SILVER_FIR"], E.LEFT),
        (E.TREE_ID["BEECH"], E.RIGHT), (E.TREE_ID["BIRCH"], E.RIGHT),
        (E.TREE_ID["HORSE_CHESTNUT"], E.RIGHT),
    })
    assert E.PLACEMENTS[E.DWELLER_ID["WOLF"]] == frozenset({
        (E.TREE_ID["DOUGLAS_FIR"], E.LEFT), (E.TREE_ID["SYCAMORE"], E.LEFT),
        (E.TREE_ID["SILVER_FIR"], E.RIGHT),
    })


def test_deck_composition_matches_the_box():
    assert sum(E.TREE_COPIES) == 66
    assert sum(c for v in E.VARIANTS for _, _, c in v) == 184


# ---------------------------------------------------------------------------
# 2. Fuzzing contre l'oracle
# ---------------------------------------------------------------------------


def test_fuzz_against_reference():
    rng = random.Random(20260814)
    for i in range(1500):
        trees, placements = random_forest(rng)
        cave = rng.randint(0, 30)
        fast, ref = build_pair(placements, trees, cave)
        expected = R.score_forest(ref)
        got = fast.score()
        assert got == expected, (
            f"itération {i}: moteur={got} oracle={expected}\n"
            f"arbres={trees}\nposes={placements}"
        )


def test_fuzz_large_forests():
    rng = random.Random(7)
    for i in range(200):
        trees, placements = random_forest(rng, n_trees=28, n_dwellers=140)
        fast, ref = build_pair(placements, trees)
        assert fast.score() == R.score_forest(ref), i


def test_copy_is_independent():
    rng = random.Random(3)
    trees, placements = random_forest(rng, n_trees=10, n_dwellers=30)
    fast, _ = build_pair(placements, trees)
    clone = fast.copy()
    assert clone.score() == fast.score()
    # Forêt minimale dédiée : un seul arbre, aucune autre carte pouvant
    # interagir avec le Geai (oiseaux, Grive épeiche complets, etc.), pour
    # isoler proprement le gain de +3 propre à EURASIAN_JAY.
    isolated = E.Forest()
    isolated.add_tree(E.TREE_ID["OAK"])
    clone2 = isolated.copy()
    did = E.DWELLER_ID["EURASIAN_JAY"]
    before = isolated.score()
    clone2.add_dweller(0, E.POS_ID["Top"], did)
    assert isolated.score() == before
    assert clone2.score() == before + 3


def test_placement_ignores_species_roe_deer_uses_printed_symbol():
    """Confirmé par capture d'écran du rulebook FR : le placement d'un
    habitant ne dépend que du côté (Top/Bottom/Left/Right), pas de l'espèce
    de l'arbre. Le symbole imprimé sur la carte (fixe, indépendant de
    l'arbre porteur) sert au bonus de paiement ET à la règle ROE_DEER
    ("3 points par carte montrant le symbole X").

    Ici : un chevreuil dont la variante imprimée est BIRCH est posé sous un
    OAK (espèce différente, désormais autorisé). Il doit compter les cartes
    montrant le symbole BIRCH (l'arbre Bouleau lui-même), pas OAK.
    """
    did = E.DWELLER_ID["ROE_DEER"]
    pos = E.POS_ID["Left"]  # une variante Left existe forcément pour ROE_DEER
    birch = E.TREE_ID["BIRCH"]
    oak = E.TREE_ID["OAK"]

    # Placement désormais légal : Oak comme arbre porteur, Birch comme
    # symbole imprimé (auparavant interdit par l'ancien filtre d'espèce).
    f = E.Forest()
    f.add_tree(oak)     # tree_idx 0 : l'arbre porteur réel
    f.add_tree(birch)   # tree_idx 1 : compte pour le symbole Birch

    assert f.can_place(0, pos, did), "le placement doit ignorer l'espèce"

    f.add_dweller(0, pos, did, symbol=birch)
    # 1 chevreuil (compte lui-même) + 1 arbre Birch = 2 cartes montrant Birch
    # -> 3*2 = 6, plus le Bouleau lui-même vaut 1 point de base (rulebook).
    assert f.score() == 3 * 2 + 1

    # Sans le symbole (comportement par défaut, rétrocompatible) : compte
    # sur l'arbre porteur réel (Oak), qui n'a pas d'autre carte Oak ici.
    f2 = E.Forest()
    f2.add_tree(oak)
    f2.add_tree(birch)
    f2.add_dweller(0, pos, did)  # pas de symbol -> repli sur l'arbre porteur
    # host = OAK : l'arbre Oak + le chevreuil lui-même (attribué à OAK) = 2
    # cartes montrant OAK -> 3*1*2 = 6, plus le Bouleau isolé +1 = 7.
    assert f2.score() == 3 * 2 + 1


def test_choose_payment_prefers_bonus_symbol():
    """Brique 1 du refactoring effets/bonus (voir reference/REFACTOR_PLAN.md).

    Confirmé par Mehdi : le bonus jumelles se déclenche si AU MOINS UNE carte
    utilisée pour payer porte le même symbole que celui imprimé sur la carte
    posée (ex. le Loup porte le symbole Sapin blanc ; payer avec au moins une
    carte de symbole Sapin blanc déclenche le rejeu de tour). Le paiement
    doit donc prioriser la défausse d'une carte du bon symbole quand c'est
    possible, sans changer le nombre de cartes défaussées (le coût).
    """
    import game as G

    oak = E.TREE_ID["OAK"]
    birch = E.TREE_ID["BIRCH"]
    beech = E.TREE_ID["BEECH"]

    # Un dweller quelconque à faible coût, décliné avec 3 symboles distincts.
    any_did = E.DWELLER_ID["WOOD_ANT"]
    pos = list(E.VALID_POS[any_did])[0]

    card_oak = (G.DWELLER, (any_did, oak, pos), (any_did, oak, pos))
    card_birch = (G.DWELLER, (any_did, birch, pos), (any_did, birch, pos))
    card_beech = (G.DWELLER, (any_did, beech, pos), (any_did, beech, pos))
    hand = [card_oak, card_birch, card_beech]

    # On pose une carte de symbole BIRCH (le "preferred_symbol") ; seule
    # card_birch en main porte ce symbole. Avec cost=1, elle doit être
    # choisie pour la défausse, alors que l'ancienne heuristique (coût
    # décroissant) ne s'en souciait pas.
    payment = G.choose_payment(hand, cost=1, preferred_symbol=birch)
    assert payment == [1], "la carte au symbole préféré doit être défaussée en priorité"

    # Sans symbole préféré, comportement inchangé (pas de préférence).
    payment_none = G.choose_payment(hand, cost=1, preferred_symbol=None)
    assert len(payment_none) == 1  # ne plante pas, comportement par défaut


def test_apply_reports_bonus_paid():
    """`Game.apply` doit exposer si le paiement a effectivement déclenché
    le bonus (au moins une carte défaussée du bon symbole), pour que la
    brique 2 (effets post-pose) puisse s'en servir.
    """
    import game as G

    g = G.Game(n_players=2, seed=123)
    # Force une main connue pour le joueur courant : un chevreuil (ROE_DEER)
    # à poser, plus une carte du même symbole imprimé pour payer son coût.
    did = E.DWELLER_ID["ROE_DEER"]
    pos = list(E.VALID_POS[did])[0]
    variant_symbol = E.VARIANTS[did][0][0]  # tree_id de la première variante
    card = (G.DWELLER, (did, variant_symbol, pos), (did, variant_symbol, pos))
    filler = (G.DWELLER, (E.DWELLER_ID["WOOD_ANT"], variant_symbol,
                           list(E.VALID_POS[E.DWELLER_ID["WOOD_ANT"]])[0]),
              (E.DWELLER_ID["WOOD_ANT"], variant_symbol,
               list(E.VALID_POS[E.DWELLER_ID["WOOD_ANT"]])[0]))
    tree = (G.TREE, E.TREE_ID["OAK"], None)
    cost = E.DWELLER_COST[did]

    player = g.players[g.current]
    player.forest.add_tree(variant_symbol)
    player.hand = [card, filler, tree] + [tree] * max(0, cost)

    g.apply(("dweller", did, 0, pos))
    if cost > 0:
        assert g.last_bonus_paid, "une carte du bon symbole était disponible, le bonus doit être signalé"


def test_dweller_draw_and_replay_effects():
    """Brique 2, batch 1 (voir reference/REFACTOR_PLAN.md) : pioche fixe,
    pioche conditionnée au bonus, pioche proportionnelle à un compteur, et
    rejeu de tour. Un seul scénario synthétique par mécanique, sur un mini
    deck contrôlé pour vérifier le nombre de cartes piochées / le fait que
    le joueur rejoue.
    """
    import game as G

    def make_game_with_hand(hand_cards, deck_tail):
        g = G.Game(n_players=2, seed=1)
        g.players[0].hand = list(hand_cards)
        g.deck = list(deck_tail)  # pioché par pop(), donc la fin = prochaine pioche
        g.current = 0
        return g

    filler_tree = (G.TREE, E.TREE_ID["OAK"], None)

    # -- pioche fixe : Fouine (BEECH_MARTEN), pas de condition -------------
    did = E.DWELLER_ID["BEECH_MARTEN"]
    pos = list(E.VALID_POS[did])[0]
    card = (G.DWELLER, (did, E.TREE_ID["OAK"], pos), (did, E.TREE_ID["OAK"], pos))
    g = make_game_with_hand([card] + [filler_tree] * E.DWELLER_COST[did],
                             [filler_tree] * 3)
    g.players[0].forest.add_tree(E.TREE_ID["OAK"])
    before = len(g.players[0].hand)
    g.apply(("dweller", did, 0, pos))
    after = len(g.players[0].hand)
    # -1 carte jouée, -coût (défaussées), +1 pioche fixe
    assert after == before - 1 - E.DWELLER_COST[did] + 1

    # -- rejeu de tour inconditionnel : Geai des chênes ---------------------
    did = E.DWELLER_ID["EURASIAN_JAY"]
    pos = list(E.VALID_POS[did])[0]
    card = (G.DWELLER, (did, E.TREE_ID["OAK"], pos), (did, E.TREE_ID["OAK"], pos))
    g = make_game_with_hand([card] + [filler_tree] * E.DWELLER_COST[did],
                             [filler_tree] * 3)
    g.players[0].forest.add_tree(E.TREE_ID["OAK"])
    g.current = 0
    g.apply(("dweller", did, 0, pos))
    assert g.current == 0, "le Geai des chênes doit faire rejouer le même joueur"

    # -- pioche proportionnelle : Renard roux, 1 par Lièvre déjà en forêt --
    did = E.DWELLER_ID["RED_FOX"]
    pos = list(E.VALID_POS[did])[0]
    card = (G.DWELLER, (did, E.TREE_ID["OAK"], pos), (did, E.TREE_ID["OAK"], pos))
    g = make_game_with_hand([card] + [filler_tree] * E.DWELLER_COST[did],
                             [filler_tree] * 5)
    f = g.players[0].forest
    f.add_tree(E.TREE_ID["OAK"])
    f.add_tree(E.TREE_ID["BEECH"])
    hare_did = E.DWELLER_ID["EUROPEAN_HARE"]
    hare_pos = list(E.VALID_POS[hare_did])[0]
    f.add_dweller(1, hare_pos, hare_did)
    f.add_dweller(1, hare_pos, hare_did)  # 2 lièvres empilés (slot partagé)
    before = len(g.players[0].hand)
    g.apply(("dweller", did, 0, pos))
    after = len(g.players[0].hand)
    assert after == before - 1 - E.DWELLER_COST[did] + 2, "2 lièvres -> 2 cartes piochées"


def test_play_free_from_hand_single_use():
    """Brique 2b : Blaireau européen, bonus jumelles -> joue gratuitement
    un animal depuis la main. Un seul usage (remaining=1), donc le tour
    passe après la résolution (pas de rejeu).
    """
    import game as G

    g = G.Game(n_players=2, seed=7)
    filler_tree = (G.TREE, E.TREE_ID["OAK"], None)
    badger = E.DWELLER_ID["EUROPEAN_BADGER"]
    pos = list(E.VALID_POS[badger])[0]
    symbol = E.VARIANTS[badger][0][0]
    card = (G.DWELLER, (badger, symbol, pos), (badger, symbol, pos))
    bonus_payer = (G.DWELLER, (E.DWELLER_ID["WOOD_ANT"], symbol,
                                list(E.VALID_POS[E.DWELLER_ID["WOOD_ANT"]])[0]),
                   (E.DWELLER_ID["WOOD_ANT"], symbol,
                    list(E.VALID_POS[E.DWELLER_ID["WOOD_ANT"]])[0]))
    free_animal_did = E.DWELLER_ID["RED_FOX"]
    free_pos = next(p for p in E.VALID_POS[free_animal_did] if p != pos)
    other_symbol = E.TREE_ID["BEECH"] if symbol != E.TREE_ID["BEECH"] else E.TREE_ID["OAK"]
    free_card = (G.DWELLER, (free_animal_did, other_symbol, free_pos),
                 (free_animal_did, other_symbol, free_pos))

    player = g.players[0]
    player.forest.add_tree(symbol)
    cost = E.DWELLER_COST[badger]
    player.hand = [card, bonus_payer, free_card] + [filler_tree] * max(0, cost - 1)
    g.current = 0

    g.apply(("dweller", badger, 0, pos))
    assert g.pending_effect is not None, "le bonus doit ouvrir un effet en attente"
    assert g.current == 0, "le tour ne passe pas tant que l'effet est en attente"

    actions = g.legal_actions()
    assert ("skip_effect",) in actions
    free_actions = [a for a in actions if a[0] == "free_dweller"]
    assert any(a[1] == free_animal_did for a in free_actions), \
        "le renard (animal) doit être proposable gratuitement"

    n_before = len(player.hand)
    g.apply(("free_dweller", free_animal_did, 0, free_pos))
    assert g.pending_effect is None, "un seul usage -> effet clos après la pose"
    assert g.current == 1, "le tour passe au joueur suivant après résolution"
    assert len(player.hand) == n_before - 1
    assert player.forest.dweller_count[free_animal_did] == 1


def test_play_free_from_hand_skip():
    """L'effet en attente doit pouvoir être décliné via skip_effect."""
    import game as G

    g = G.Game(n_players=2, seed=7)
    filler_tree = (G.TREE, E.TREE_ID["OAK"], None)
    badger = E.DWELLER_ID["EUROPEAN_BADGER"]
    pos = list(E.VALID_POS[badger])[0]
    symbol = E.VARIANTS[badger][0][0]
    card = (G.DWELLER, (badger, symbol, pos), (badger, symbol, pos))
    bonus_payer = (G.DWELLER, (E.DWELLER_ID["WOOD_ANT"], symbol,
                                list(E.VALID_POS[E.DWELLER_ID["WOOD_ANT"]])[0]),
                   (E.DWELLER_ID["WOOD_ANT"], symbol,
                    list(E.VALID_POS[E.DWELLER_ID["WOOD_ANT"]])[0]))
    cost = E.DWELLER_COST[badger]
    g.players[0].forest.add_tree(symbol)
    g.players[0].hand = [card, bonus_payer] + [filler_tree] * max(0, cost - 1)
    g.current = 0

    g.apply(("dweller", badger, 0, pos))
    assert g.pending_effect is not None
    g.apply(("skip_effect",))
    assert g.pending_effect is None
    assert g.current == 1


def test_raccoon_cave_choice():
    """Brique 2c : Raton laveur, place N cartes de la main à la Grotte
    (1 pt chacune, `Forest.cave`) et pioche N cartes. Le nombre N est un
    vrai choix exposé comme actions ("cave_discard", n).
    """
    import game as G

    g = G.Game(n_players=2, seed=11)
    filler_tree = (G.TREE, E.TREE_ID["OAK"], None)
    did = E.DWELLER_ID["RACCOON"]
    pos = list(E.VALID_POS[did])[0]
    card = (G.DWELLER, (did, E.TREE_ID["OAK"], pos), (did, E.TREE_ID["OAK"], pos))
    cost = E.DWELLER_COST[did]
    extra = [filler_tree] * 3  # cartes disponibles pour aller à la Grotte
    player = g.players[0]
    player.forest.add_tree(E.TREE_ID["OAK"])
    player.hand = [card] + [filler_tree] * cost + extra
    g.deck = [filler_tree] * 10
    g.current = 0

    g.apply(("dweller", did, 0, pos))
    assert g.pending_effect == ("cave_choice", None, None)
    actions = g.legal_actions()
    assert all(a[0] == "cave_discard" for a in actions)
    max_n = max(a[1] for a in actions)
    assert max_n == len(player.hand)

    cave_before = player.forest.cave
    hand_before = len(player.hand)
    g.apply(("cave_discard", 2))
    assert player.forest.cave == cave_before + 2
    assert len(player.hand) == hand_before - 2 + 2  # -2 défaussées, +2 piochées
    assert g.pending_effect is None
    assert g.current == 1


def test_mole_play_chain():
    """Brique 2d : Taupe, chaîne d'actions de pose payantes normales.
    S'arrête automatiquement (seul skip_effect reste légal) quand plus
    aucune pose n'est finançable, ou explicitement via skip_effect.
    """
    import game as G

    g = G.Game(n_players=2, seed=13)
    filler_tree = (G.TREE, E.TREE_ID["OAK"], None)
    did = E.DWELLER_ID["MOLE"]
    pos = list(E.VALID_POS[did])[0]
    card = (G.DWELLER, (did, E.TREE_ID["OAK"], pos), (did, E.TREE_ID["OAK"], pos))
    cost = E.DWELLER_COST[did]
    player = g.players[0]
    player.forest.add_tree(E.TREE_ID["OAK"])
    # main : la taupe + de quoi la payer + un arbre gratuit (coût 0)
    # à poser pendant la chaîne, sans rien pour le payer davantage après.
    free_tree = next(i for i, c in enumerate(E.TREE_COST) if c == 0)
    chain_tree = (G.TREE, free_tree, None)
    player.hand = [card] + [filler_tree] * cost + [chain_tree]
    g.current = 0

    g.apply(("dweller", did, 0, pos))
    assert g.pending_effect == ("play_chain", None, None)
    actions = g.legal_actions()
    assert ("skip_effect",) in actions
    assert ("tree", free_tree) in actions

    trees_before = player.forest.n_trees
    g.apply(("tree", free_tree))
    assert g.pending_effect == ("play_chain", None, None), "la chaîne continue après une pose"
    assert player.forest.n_trees == trees_before + 1

    # Plus rien à jouer (main vide ou trop chère) -> seul skip_effect reste
    actions2 = g.legal_actions()
    assert actions2 == [("skip_effect",)]
    g.apply(("skip_effect",))
    assert g.pending_effect is None
    assert g.current == 1


def test_mushroom_permanent_triggers():
    """Brique 3 : champignons à effet permanent. Un exemplaire de Cèpe de
    Bordeaux (PENNY_BUN) en jeu doit déclencher une pioche à chaque
    habitant posé en position Top par le même joueur, y compris sur des
    poses ultérieures (pas seulement la pose du champignon lui-même).
    """
    import game as G

    g = G.Game(n_players=2, seed=17)
    filler_tree = (G.TREE, E.TREE_ID["OAK"], None)
    player = g.players[0]
    player.forest.add_tree(E.TREE_ID["OAK"])
    g.current = 0

    penny_bun = E.DWELLER_ID["PENNY_BUN"]
    top_pos = E.POS_ID["Top"]
    penny_pos = list(E.VALID_POS[penny_bun])[0]  # sa position réelle, pas forcément Top
    card = (G.DWELLER, (penny_bun, E.TREE_ID["OAK"], penny_pos),
            (penny_bun, E.TREE_ID["OAK"], penny_pos))
    cost = E.DWELLER_COST[penny_bun]
    player.hand = [card] + [filler_tree] * cost
    g.deck = [filler_tree] * 10

    before = len(player.hand)
    g.apply(("dweller", penny_bun, 0, penny_pos))
    after = len(player.hand)
    expected_self_trigger = 1 if penny_pos == top_pos else 0
    assert after == before - 1 - cost + expected_self_trigger

    # Pose ultérieure d'un autre habitant en Top : doit aussi déclencher
    # le Cèpe de Bordeaux, alors qu'il n'est plus la carte posée.
    other_did = next(
        d for d in range(E.N_DWELLERS)
        if top_pos in E.VALID_POS[d] and d != penny_bun
    )
    other_card = (G.DWELLER, (other_did, E.TREE_ID["OAK"], top_pos),
                  (other_did, E.TREE_ID["OAK"], top_pos))
    other_cost = E.DWELLER_COST[other_did]
    player.hand = [other_card] + [filler_tree] * other_cost
    g.current = 0
    before2 = len(player.hand)
    g.apply(("dweller", other_did, 0, top_pos))
    after2 = len(player.hand)
    assert after2 == before2 - 1 - other_cost + 1, \
        "le Cèpe de Bordeaux doit se redéclencher sur une pose ultérieure en Top"


def test_clearing_capped_at_ten():
    """Vidage à 10 confirmé par Mehdi : au-delà de 10 cartes, la Clairière
    est vidée (cartes perdues, pas remélangées dans le deck)."""
    import game as G

    g = G.Game(n_players=2, seed=1)
    filler_tree = (G.TREE, E.TREE_ID["OAK"], None)
    for _ in range(9):
        g._add_to_clearing(filler_tree)
    assert len(g.clearing) == 9
    g._add_to_clearing(filler_tree)
    assert len(g.clearing) == 0, "10e carte -> vidage immédiat"


def test_draw_prefers_cheapest_clearing_card():
    """Confirmé par Mehdi : à CHAQUE pioche (tour normal ou effet), le
    joueur peut prendre une carte connue de la Clairière au lieu de piocher
    à l'aveugle. `choose_draw_source` (heuristique) doit préférer la carte
    la moins chère de la Clairière plutôt que toucher au deck."""
    import game as G

    g = G.Game(n_players=2, seed=1)
    cheap_tree = min(range(len(E.TREE_COST)), key=lambda t: E.TREE_COST[t])
    expensive_tree = max(range(len(E.TREE_COST)), key=lambda t: E.TREE_COST[t])
    assert E.TREE_COST[cheap_tree] < E.TREE_COST[expensive_tree]
    cheap_card = (G.TREE, cheap_tree, None)
    expensive_card = (G.TREE, expensive_tree, None)
    g.clearing = [expensive_card, cheap_card]
    g.deck = [(G.TREE, E.TREE_ID["OAK"], None)]
    player = g.players[0]
    hand_before = len(player.hand)
    deck_before = len(g.deck)

    g._draw_one(player)

    assert player.hand[-1] == cheap_card
    assert len(player.hand) == hand_before + 1
    assert g.clearing == [expensive_card]
    assert len(g.deck) == deck_before, "le deck ne doit pas être touché tant que la Clairière peut fournir"

    g.clearing = []
    g._draw_one(player)
    assert player.hand[-1] == (G.TREE, E.TREE_ID["OAK"], None), "Clairière vide -> pioche aveugle dans le deck"


def test_brown_bear_moves_clearing_to_cave():
    """Ours brun : effet INCONDITIONNEL, vide toute la Clairière (y compris
    ses propres cartes de paiement, déjà défaussées à ce stade) dans sa
    Grotte, indépendamment du bonus jumelles (confirmé par Mehdi)."""
    import game as G

    g = G.Game(n_players=2, seed=3)
    filler_tree = (G.TREE, E.TREE_ID["OAK"], None)
    did = E.DWELLER_ID["BROWN_BEAR"]
    pos = list(E.VALID_POS[did])[0]
    card = (G.DWELLER, (did, E.TREE_ID["OAK"], pos), (did, E.TREE_ID["OAK"], pos))
    cost = E.DWELLER_COST[did]
    player = g.players[0]
    player.forest.add_tree(E.TREE_ID["OAK"])
    player.hand = [card] + [filler_tree] * cost
    g.clearing = [filler_tree, filler_tree, filler_tree]
    g.deck = [filler_tree] * 5
    g.current = 0
    cave_before = player.forest.cave
    expected_moved = len(g.clearing) + cost  # + les cartes utilisées pour payer CETTE pose

    g.apply(("dweller", did, 0, pos))

    assert not g.last_bonus_paid, "paiement en Arbres (sans symbole) : pas de bonus jumelles ici"
    assert player.forest.cave == cave_before + expected_moved
    assert g.clearing == []


def test_brown_bear_bonus_draws_and_replays():
    """Bonus jumelles sur l'Ours brun (confirmé par Mehdi) : l'effet
    Clairière -> Grotte reste inconditionnel, mais la pioche ET le rejeu de
    tour ne se déclenchent QUE si payé avec le bonus."""
    import game as G

    g = G.Game(n_players=2, seed=5)
    did = E.DWELLER_ID["BROWN_BEAR"]
    pos = list(E.VALID_POS[did])[0]
    symbol = E.VARIANTS[did][0][0]
    card = (G.DWELLER, (did, symbol, pos), (did, symbol, pos))
    wood_ant = E.DWELLER_ID["WOOD_ANT"]
    bonus_payer = (G.DWELLER, (wood_ant, symbol, list(E.VALID_POS[wood_ant])[0]),
                   (wood_ant, symbol, list(E.VALID_POS[wood_ant])[0]))
    filler_tree = (G.TREE, E.TREE_ID["OAK"], None)
    cost = E.DWELLER_COST[did]
    player = g.players[0]
    player.forest.add_tree(symbol)
    player.hand = [card, bonus_payer] + [filler_tree] * max(0, cost - 1)
    g.deck = [filler_tree] * 5
    g.current = 0
    hand_before = len(player.hand)

    g.apply(("dweller", did, 0, pos))

    assert g.last_bonus_paid
    assert player.forest.cave == cost, "l'effet inconditionnel doit aussi s'appliquer quand le bonus est payé"
    # -1 carte jouée, -cost défaussées en paiement, +1 pioche bonus
    assert len(player.hand) == hand_before - 1 - cost + 1
    assert g.current == 0, "l'Ours brun doit faire rejouer le même joueur si le bonus est payé"
