import random
from search import greedy_action, MCTS, determinize, Node
from game import Game
from engine import TREES, DWELLERS, POSITIONS

TREE_NAMES = [t.name for t in TREES]
DWELLER_NAMES = [d.name for d in DWELLERS]

def action_label(action):
    if action[0] == "draw":
        return "Pioche"
    if action[0] == "tree":
        return f"Plante {TREE_NAMES[action[1]]}"
    _, did, tree_idx, pos = action
    return f"Pose {DWELLER_NAMES[did]} en {POSITIONS[pos]} (arbre #{tree_idx})"

seed = 5555  # une des parties MCTS déjà jouées
game = Game(n_players=2, seed=seed)
rng = random.Random(seed)
bot = MCTS(observer=0, iterations=300, seed=seed, rollout_depth=35)

turn = 0
log = []
while not game.over and turn < 200:
    me = game.current
    if me == 0:
        # snapshot des candidats avant de choisir
        root = bot.root
        action = bot.choose(game)
        alt = sorted(root.children.items(), key=lambda kv: -kv[1].visits)[:4]
        alt_str = ", ".join(f"{action_label(a)} ({c.visits}v)" for a, c in alt)
    else:
        action = greedy_action(game, rng)
        alt_str = None

    score_before = game.scores()[0]
    game.apply(action)
    bot.advance(action)
    score_after = game.scores()[0]

    if me == 0:
        turn += 1
        log.append((turn, action_label(action), score_before, score_after, alt_str))

for t, lab, sb, sa, alt in log:
    gain = sa - sb
    print(f"Tour {t:2d} | MCTS joue: {lab:45s} | score {sb:3d}->{sa:3d} (+{gain}) | candidats vus: {alt}")

print()
print("Score final:", game.scores())
