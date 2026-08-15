import random, statistics as st
from search import greedy_policy, MCTS, play_game

def mcts_policy(iterations, depth):
    def pol(game, rng):
        m = MCTS(game, observer=game.current, iterations=iterations, max_depth=depth, rng=rng)
        return m.search()
    return pol

def run_batch(policies, n, seed0=0):
    results = []
    for i in range(n):
        scores, turns = play_game(policies, n_players=2, seed=seed0 + i)
        results.append(scores)
    return results

random.seed(0)

print("=== Greedy vs Greedy, 40 parties, sieges alternes ===")
results = []
for i in range(40):
    pols = [greedy_policy, greedy_policy] if i % 2 == 0 else [greedy_policy, greedy_policy]
    scores, turns = play_game(pols, seed=1000+i)
    results.append(scores)
p0 = [s[0] for s in results]
p1 = [s[1] for s in results]
print("P0 mean", st.mean(p0), "stdev", st.pstdev(p0))
print("P1 mean", st.mean(p1), "stdev", st.pstdev(p1))
diff = [a-b for a,b in zip(p0,p1)]
print("diff mean", st.mean(diff), "stdev", st.pstdev(diff))
