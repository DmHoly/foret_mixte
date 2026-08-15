import time, random, statistics as st
from search import greedy_action, MCTS
from game import Game

def one_game(seed, me, iterations=400, depth=40):
    game = Game(n_players=2, seed=seed)
    rng = random.Random(seed)
    bot = MCTS(observer=me, iterations=iterations, seed=seed, rollout_depth=depth)
    turns = 0
    while not game.over and turns < 600:
        if game.current == me:
            action = bot.choose(game)
        else:
            action = greedy_action(game, rng)
        game.apply(action)
        bot.advance(action)
        turns += 1
    return game.scores()

N = 10
mcts_scores, greedy_scores, diffs = [], [], []
t0 = time.time()
for s in range(N):
    me = s % 2
    sc = one_game(6000 + s, me)
    mcts_scores.append(sc[me])
    greedy_scores.append(sc[1-me])
    diffs.append(sc[me] - sc[1-me])
print(f"{time.time()-t0:.0f}s")
print("MCTS scores:", mcts_scores)
print("Greedy scores:", greedy_scores)
print("MCTS mean", st.mean(mcts_scores), "stdev", st.pstdev(mcts_scores))
print("Greedy mean", st.mean(greedy_scores), "stdev", st.pstdev(greedy_scores))
print("diff mean", st.mean(diffs), "stdev", st.pstdev(diffs))
print("wins", sum(1 for d in diffs if d>0), "/", N)
