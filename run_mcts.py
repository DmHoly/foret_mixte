import time, random, statistics as st
from search import greedy_policy, MCTS, play_game

def mcts_policy(iterations, depth):
    def pol(game, rng):
        m = MCTS(game, observer=game.current, iterations=iterations, max_depth=depth, rng=rng)
        return m.search()
    return pol

N = 16
t0 = time.time()
results = []
for i in range(N):
    if i % 2 == 0:
        pols = [mcts_policy(300, 35), greedy_policy]
        mcts_seat = 0
    else:
        pols = [greedy_policy, mcts_policy(300, 35)]
        mcts_seat = 1
    scores, turns = play_game(pols, seed=2000+i)
    results.append((scores[mcts_seat], scores[1-mcts_seat], scores[mcts_seat]-scores[1-mcts_seat]))
t1 = time.time()

mcts_scores = [r[0] for r in results]
greedy_scores = [r[1] for r in results]
diffs = [r[2] for r in results]
wins = sum(1 for d in diffs if d > 0)
print(f"N={N} temps={t1-t0:.1f}s ({(t1-t0)/N:.2f}s/partie)")
print("MCTS  mean", st.mean(mcts_scores), "stdev", st.pstdev(mcts_scores))
print("Greedy mean", st.mean(greedy_scores), "stdev", st.pstdev(greedy_scores))
print("diff mean", st.mean(diffs), "stdev", st.pstdev(diffs))
print("victoires MCTS:", wins, "/", N)
