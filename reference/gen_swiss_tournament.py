"""Tournoi Swiss entre les bots du depot, pour illustrer la progression du
projet dans le README (greedy nu -> MCTS -> heuristiques qui battent MCTS ->
tiebreak Gradient Boosting).

5 participants : BEAM (recherche en faisceau, perd contre greedy des le
depart), B (greedy + heuristiques, longtemps le meilleur bot), D (MCTS
150it, sans tiebreak), E (greedy + tiebreak GBM, le bot le plus fort
actuel), F (MCTS 150it + tiebreak GBM). Memes definitions que
`play_vs_bot.py` (B/D/E/F reutilises directement, BEAM ajoute
`search.beam_policy`).

Appariement Swiss standard : classement courant (points, puis ecart de
score cumule en tie-break), appariement des adversaires adjacents sans
repetition quand possible, "bye" (victoire automatique) tournant si nombre
impair de joueurs. Chaque match = 2 parties (sieges alternes), 1 point de
match au vainqueur (majorite des parties), 0.5/0.5 en cas d'egalite.

Usage : python reference/gen_swiss_tournament.py [n_rounds] [games_per_match]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from game import Game
from search import beam_policy
from play_vs_bot import make_bot


class BeamBot:
    def choose(self, game):
        import random
        return beam_policy(game, random.Random(), width=4, depth=2)

    def on_move_applied(self, action):
        pass


def make_any_bot(name, seat, seed):
    if name == "BEAM":
        return BeamBot()
    return make_bot(name, seat, seed)


def play_one(name_a, name_b, seat_a, seed):
    game = Game(n_players=2, seed=seed)
    seat_b = 1 - seat_a
    bots = {seat_a: make_any_bot(name_a, seat_a, seed),
            seat_b: make_any_bot(name_b, seat_b, seed)}
    turns = 0
    while not game.over and turns < 600:
        cur = game.current
        action = bots[cur].choose(game)
        game.apply(action)
        bots[cur].on_move_applied(action)
        turns += 1
    scores = game.scores()
    return scores[seat_a], scores[seat_b]


def run_swiss(players, n_rounds, games_per_match, seed_base=90000):
    standings = {p: {"points": 0.0, "margin": 0, "opponents": set(), "byes": 0}
                 for p in players}
    log = []
    seed_ctr = 0

    for rnd in range(1, n_rounds + 1):
        order = sorted(players, key=lambda p: (-standings[p]["points"], -standings[p]["margin"]))

        bye = None
        pool = order[:]
        if len(pool) % 2 == 1:
            for name in reversed(order):
                if standings[name]["byes"] == 0:
                    bye = name
                    break
            if bye is None:
                bye = order[-1]
            pool = [p for p in pool if p != bye]

        pairs = []
        while pool:
            a = pool.pop(0)
            match = None
            for i, b in enumerate(pool):
                if b not in standings[a]["opponents"]:
                    match = pool.pop(i)
                    break
            if match is None:
                match = pool.pop(0)
            pairs.append((a, match))

        if bye is not None:
            standings[bye]["points"] += 1.0
            standings[bye]["byes"] += 1
            log.append((rnd, bye, None, None, None, None))

        for a, b in pairs:
            a_wins = b_wins = 0
            margin = 0
            for g in range(games_per_match):
                seed_ctr += 1
                seed = seed_base + seed_ctr
                seat_a = g % 2
                sa, sb = play_one(a, b, seat_a, seed)
                margin += sa - sb
                if sa > sb:
                    a_wins += 1
                elif sb > sa:
                    b_wins += 1
            standings[a]["opponents"].add(b)
            standings[b]["opponents"].add(a)
            standings[a]["margin"] += margin
            standings[b]["margin"] -= margin
            if a_wins > b_wins:
                standings[a]["points"] += 1.0
            elif b_wins > a_wins:
                standings[b]["points"] += 1.0
            else:
                standings[a]["points"] += 0.5
                standings[b]["points"] += 0.5
            log.append((rnd, a, b, a_wins, b_wins, margin))

    final_order = sorted(players, key=lambda p: (-standings[p]["points"], -standings[p]["margin"]))
    return final_order, standings, log


def main():
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    games_per_match = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    players = ["BEAM", "B", "D", "E", "F"]

    t0 = time.time()
    final_order, standings, log = run_swiss(players, n_rounds, games_per_match)

    print(f"Tournoi Swiss : {len(players)} bots, {n_rounds} rondes, "
          f"{games_per_match} parties/match -- {time.time()-t0:.0f}s\n")

    print("=== Journal des rondes ===")
    for rnd, a, b, aw, bw, margin in log:
        if b is None:
            print(f"  Ronde {rnd} : {a} -- bye (+1)")
        else:
            print(f"  Ronde {rnd} : {a} vs {b} -- {aw}-{bw} (marge {margin:+d})")

    print("\n=== Classement final ===")
    print(f"{'#':>2s} {'Bot':6s} {'Points':>7s} {'Marge cumulee':>14s}")
    for i, p in enumerate(final_order, 1):
        s = standings[p]
        print(f"{i:2d} {p:6s} {s['points']:7.1f} {s['margin']:14d}")


if __name__ == "__main__":
    main()
