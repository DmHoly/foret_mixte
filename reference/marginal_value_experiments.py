"""
Valeur marginale ISOLEE de plusieurs mecaniques, methode "common random
numbers" (meme technique que gen_pairwise_dataset.py) : on clone la
partie a un instant T, on applique UNE SEULE intervention differente sur
chaque clone, puis les deux continuent avec la MEME politique (greedy,
epsilon=0 -> deterministe, donc aucun tirage aleatoire ne fait diverger
les deux branches independamment du deck lui-meme -- Game.clone() copie
la liste deck telle quelle). Le score final differe UNIQUEMENT a cause de
l'intervention et de ses consequences en cascade, pas de bruit de tirage
independant entre les deux branches.

Session du 16/08 (sur demande de Mehdi), trois questions :

1. Empiler des Lievres sur un Sapin blanc plutot qu'ailleurs : combo
   additif exact (pas besoin de simuler, c'est une formule) -- +2 pts par
   habitant empile sur un Sapin blanc, en plus du carre du Lievre.

2. Valeur marginale d'un rejeu de tour ISOLE (Geai des chenes,
   REPLAY_ALWAYS inconditionnel -- cas le plus propre, pas de dependance
   a un bonus jumelles paye ou non). Mesure a la fois le score absolu du
   joueur et le DIFFERENTIEL (joueur0 - joueur1), pour trancher "je gagne
   au moins sur l'ecart avec l'adversaire, meme si mon score absolu ne
   bouge pas".

3. Valeur marginale d'UNE carte de plus en main, aveugle (deck) vs connue
   et forte (proxy pour un choix cible en Clairiere) : deck INCHANGE dans
   les deux clones (la carte est dupliquee, pas piochee), pour ne pas
   decaler l'ordre de pioche du reste de la partie -- sinon le bruit induit
   noie completement le signal (constate empiriquement, cf. historique).

Usage :
    python reference/marginal_value_experiments.py [n_seeds]
"""
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import engine as E  # noqa: E402
from game import DWELLER, TREE, Game  # noqa: E402
from search import greedy_action  # noqa: E402

JAY_DID = E.DWELLER_ID["EURASIAN_JAY"]
SYCAMORE_CARD = (TREE, E.TREE_ID["SYCAMORE"], None)


# ---------------------------------------------------------------------------
# 1. Sapin blanc + Lievre : formule exacte, pas de simulation necessaire
# ---------------------------------------------------------------------------

def silver_fir_hare_stacking():
    def score_stack(host_tree_name, n_hares):
        f = E.Forest()
        i = f.add_tree(E.TREE_ID[host_tree_name])
        hare = E.DWELLER_ID["EUROPEAN_HARE"]
        pos = list(E.VALID_POS[hare])[0]
        for _ in range(n_hares):
            f.add_dweller(i, pos, hare)
        return f.score()

    print("N lievres | sur Sapin blanc | ailleurs (Hetre) | bonus du Sapin")
    for n in range(1, 8):
        on_fir = score_stack("SILVER_FIR", n)
        elsewhere = score_stack("BEECH", n)
        print(f"{n:9d} | {on_fir:15d} | {elsewhere:17d} | +{on_fir - elsewhere}")


# ---------------------------------------------------------------------------
# 2. Rejeu isole (Geai des chenes)
# ---------------------------------------------------------------------------

def play_until_jay_playable(seed, max_turns=400):
    game = Game(n_players=2, seed=seed)
    turns = 0
    while not game.over and turns < max_turns:
        if game.current == 0:
            actions = game.legal_actions()
            jay_action = next((a for a in actions
                                if a[0] == "dweller" and a[1] == JAY_DID), None)
            if jay_action is not None:
                return game, jay_action
        game.apply(greedy_action(game, None))
        turns += 1
    return None, None


def finish_naturally(game, max_turns=1000):
    turns = 0
    while not game.over and turns < max_turns:
        game.apply(greedy_action(game, None))
        turns += 1
    return game


def _report(label, diffs):
    if len(diffs) < 2:
        print(f"{label} : pas assez d'echantillons (n={len(diffs)})")
        return
    se = statistics.stdev(diffs) / (len(diffs) ** 0.5)
    print(f"{label} : {statistics.mean(diffs):+.1f} pts (n={len(diffs)}, erreur-type={se:.1f})")


def experiment_replay(seeds):
    abs_all, abs_div, diff_all, diff_div = [], [], [], []
    for seed in seeds:
        game, jay_action = play_until_jay_playable(seed)
        if game is None:
            continue
        natural_action = greedy_action(game, None)

        clone_a = game.clone()
        clone_a.apply(natural_action)
        finish_naturally(clone_a)
        sa0, sa1 = clone_a.scores()

        clone_b = game.clone()
        clone_b.apply(jay_action)
        finish_naturally(clone_b)
        sb0, sb1 = clone_b.scores()

        abs_diff = sb0 - sa0
        margin_diff = (sb0 - sb1) - (sa0 - sa1)
        abs_all.append(abs_diff)
        diff_all.append(margin_diff)
        if natural_action != jay_action:
            abs_div.append(abs_diff)
            diff_div.append(margin_diff)

    _report("Score absolu J0, tous cas", abs_all)
    _report("Score absolu J0, cas ou forcer change le choix naturel", abs_div)
    _report("Differentiel J0-J1, tous cas", diff_all)
    _report("Differentiel J0-J1, cas ou forcer change le choix naturel", diff_div)


# ---------------------------------------------------------------------------
# 3. Carte gratuite : aveugle vs connue et forte
# ---------------------------------------------------------------------------

def _next_blind_card(game):
    """Prochaine carte non-Hiver depuis le dessus du deck (WINTER=2) --
    ce que donnerait une pioche aveugle normale. Lecture seule."""
    for i in range(len(game.deck) - 1, max(-1, len(game.deck) - 6), -1):
        if game.deck[i][0] != 2:
            return game.deck[i]
    return None


def _known_good_card(game):
    """Carte CONNUE et forte (Sycomore, combo n1 du guide de combos) --
    proxy pour 'choisir dans la Clairiere plutot que piocher a l'aveugle'."""
    return SYCAMORE_CARD


def experiment_free_card(seeds, own_turn_target, pick_card, label):
    abs_diffs, margin_diffs = [], []
    for seed in seeds:
        game = Game(n_players=2, seed=seed)
        turns = 0
        own_idx = 0
        while not game.over and turns < 600:
            if game.current == 0:
                if own_idx == own_turn_target:
                    break
                own_idx += 1
            game.apply(greedy_action(game, None))
            turns += 1
        if game.over or game.current != 0 or not game.deck:
            continue

        dup_card = pick_card(game)
        if dup_card is None:
            continue

        clone_a = game.clone()
        finish_naturally(clone_a)
        sa0, sa1 = clone_a.scores()

        clone_b = game.clone()
        clone_b.players[0].hand.append(dup_card)  # deck INCHANGE, juste duplique en main
        finish_naturally(clone_b)
        sb0, sb1 = clone_b.scores()

        abs_diffs.append(sb0 - sa0)
        margin_diffs.append((sb0 - sb1) - (sa0 - sa1))

    _report(f"[{label}] Score absolu J0", abs_diffs)
    _report(f"[{label}] Differentiel J0-J1", margin_diffs)


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seeds = list(range(1, n_seeds + 1))

    print("=== 1. Empiler les Lievres sur un Sapin blanc (formule exacte) ===")
    silver_fir_hare_stacking()

    print("\n=== 2. Valeur marginale d'un rejeu isole (Geai des chenes) ===")
    experiment_replay(seeds)

    print("\n=== 3a. Valeur marginale d'une carte gratuite AVEUGLE (deck) ===")
    experiment_free_card(seeds, own_turn_target=15, pick_card=_next_blind_card, label="aveugle")

    print("\n=== 3b. Valeur marginale d'une carte gratuite CONNUE et FORTE (proxy Clairiere) ===")
    experiment_free_card(seeds, own_turn_target=15, pick_card=_known_good_card, label="Sycomore connu")
