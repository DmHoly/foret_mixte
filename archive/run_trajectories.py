import random, statistics as st, json
from search import greedy_action, MCTS
from game import Game

def greedy_trajectory(seed):
    game = Game(n_players=2, seed=seed)
    rng = random.Random(seed)
    traj = []
    my_turn = 0
    while not game.over and my_turn < 80:
        me = game.current
        action = greedy_action(game, rng)
        game.apply(action)
        if me == 0:
            my_turn += 1
            traj.append(game.scores()[0])
    return traj

def mcts_trajectory(seed, iterations=300, depth=35):
    game = Game(n_players=2, seed=seed)
    rng = random.Random(seed)
    bot = MCTS(observer=0, iterations=iterations, seed=seed, rollout_depth=depth)
    traj = []
    my_turn = 0
    while not game.over and my_turn < 80:
        me = game.current
        if me == 0:
            action = bot.choose(game)
        else:
            action = greedy_action(game, rng)
        game.apply(action)
        bot.advance(action)
        if me == 0:
            my_turn += 1
            traj.append(game.scores()[0])
    return traj

N_GREEDY = 30
N_MCTS = 8

greedy_trajs = [greedy_trajectory(3000+i) for i in range(N_GREEDY)]
print("greedy done", [len(t) for t in greedy_trajs])

mcts_trajs = [mcts_trajectory(4000+i) for i in range(N_MCTS)]
print("mcts done", [len(t) for t in mcts_trajs])

with open("/home/claude/trajs.json", "w") as f:
    json.dump({"greedy": greedy_trajs, "mcts": mcts_trajs}, f)
