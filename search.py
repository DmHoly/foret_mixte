"""
Recherche arborescente pour Forêt Mixte.

Trois politiques, de la moins chère à la plus chère :

  greedy_policy   : joue le coup qui maximise le gain de score immédiat.
  beam_policy     : lookahead de profondeur fixe sur la main, sans adversaire.
  MCTS            : arbre persistant, sélection UCT, rollouts guidés.

Choix de conception du MCTS, et pourquoi :

1. **Déterminisation par itération** (single-observer ISMCTS). Le deck et les
   mains adverses sont cachés. Chercher sur l'état réel serait de la triche et
   surestimerait la qualité des coups. À chaque itération on rebat les cartes
   invisibles, on descend l'arbre en n'autorisant que les actions légales dans
   cette déterminisation, et les statistiques s'accumulent sur un arbre unique.
   C'est le compromis standard pour un jeu à information imparfaite quand on
   n'a pas de réseau de croyance.

2. **Rollouts guidés, pas aléatoires.** Un rollout uniforme sur ~1000 actions
   légales ne dit rien d'utile. La politique de rollout est gloutonne avec une
   part d'aléatoire (epsilon), ce qui est abordable parce que le scoring coûte
   4 us : évaluer 30 coups candidats revient à 120 us.

3. **Récompense = rang normalisé, pas score brut.** Le score absolu varie d'un
   facteur 5 selon le tirage ; backpropager le score brut fait dominer la
   chance sur la décision. On backpropage la position relative du joueur.

4. **Arbre réutilisé entre les tours.** Après un coup joué, le sous-arbre
   correspondant devient la nouvelle racine. Sur une partie complète cela vaut
   un facteur 2 à 3 sur le nombre effectif de simulations par décision.

Ce que ce module ne fait PAS, et qu'il faudrait pour aller vers l'état de
l'art :
  - pas de réseau de valeur ni de politique apprise (voir REVUE.md : mctx
    attend une fonction de valeur, il ne fait pas de rollouts)
  - pas de parallélisation racine (multiprocessing) : facile à ajouter,
    volontairement laissé de côté pour garder le module lisible
  - le paiement reste une heuristique et non une décision de l'arbre
"""

import math
import random

from engine import TREE_ID
from game import DWELLER, TREE, WINTER, Game, insert_winter_cards, _is_strong_card

C_PUCT = 1.4
ROLLOUT_EPSILON = 0.25

# Prime de rollout pour le Sycomore (voir `rollout`, `greedy_action`) :
# combo n°1 par espérance dans docs/combo_guide.html (49 pts, 92% des
# parties), mais delta immédiat quasi nul à la pose -> sous-joué par un
# rollout myope sans cette correction. Valeur non calibrée finement,
# testée en tête-à-tête MCTS avant adoption (voir reference/, session du
# 15/08 sur demande de Mehdi).
SYCAMORE_ROLLOUT_BONUS = {TREE_ID["SYCAMORE"]: 10.0}
ROLLOUT_CANDIDATES = 8

# Prime d'urgence Clairière (idée de Mehdi, session du 16/08) : sans elle,
# `greedy_action` ne regarde JAMAIS la Clairière pour décider s'il joue une
# carte de la main ou pioche -- il pose dès que `best_gain > 0`, même un
# petit gain, laissant une carte forte visible en Clairière à la merci de
# l'adversaire au tour suivant. La Clairière est une ressource partagée et
# contestée (10 cartes max, débordement = perte définitive), contrairement
# au delta immédiat d'une pose qui reste acquis qu'on la joue ce tour-ci ou
# le suivant. Validé en tête-à-tête (greedy+urgence vs greedy nu, n=300,
# 300 seeds) : ~61% de victoires, effet plateau au-delà de +20/+25 pts ; et
# contre MCTS(150it) par défaut (n=30) : 19/30, écart moyen +65.5 pts.
CLEARING_URGENCY_BONUS = 25.0


def _fallback_action(game):
    """Action de repli quand aucune autre n'est retenue.

    ("draw",) n'est légale que hors `pending_effect` (brique 2b : effet
    "jouer gratuitement depuis la main" en attente) ; dans ce cas c'est
    ("skip_effect",) qui joue ce rôle.
    """
    return ("skip_effect",) if game.pending_effect is not None else ("draw",)


# ---------------------------------------------------------------------------
# Politiques simples (baselines et politique de rollout)
# ---------------------------------------------------------------------------


def greedy_action(game, rng, epsilon=0.0, candidates=None, tree_bonus=6,
                   tree_combo_bonus=None, clearing_urgency=CLEARING_URGENCY_BONUS,
                   tiebreak=None, tiebreak_margin=3.0):
    """Coup maximisant le gain immédiat, avec un correctif d'ouverture.

    Un arbre ne rapporte presque rien à la pose mais ouvre 4 emplacements.
    Sans correctif, une politique purement gloutonne ne plante jamais d'arbre
    et plafonne très bas. `tree_bonus` est une prime décroissante avec le
    nombre d'arbres déjà en forêt ; c'est une heuristique, pas une règle.

    `tree_combo_bonus` (optionnel, {tree_id: bonus}) : prime supplémentaire
    fixe à la plantation d'une espèce donnée, pour corriger un biais
    d'horizon spécifique au ROLLOUT de MCTS (voir `rollout()`) -- PAS
    activé par défaut, et volontairement absent de `greedy_policy` : une
    tentative précédente d'injecter une table de force globale dans la
    politique de décision elle-même (reference/strength_policy.py) a
    dégradé la qualité de jeu (le delta exact résiste mal à une correction
    grossière appliquée à CHAQUE décision réelle). Ici la correction ne
    s'applique qu'aux coups simulés à l'intérieur d'un rollout tronqué,
    pas aux décisions réellement jouées.

    `clearing_urgency` (voir `CLEARING_URGENCY_BONUS`) : contrairement à
    `tree_combo_bonus`, celui-ci EST actif par défaut sur les décisions
    réelles -- validé comme tel (pas seulement en rollout), voir la
    docstring de la constante. Traite "piocher pour ramasser une carte
    forte visible en Clairière" comme un candidat à gain fictif, mis en
    concurrence avec les poses de la main : sans ça, `best_gain > 0` fait
    toujours préférer poser une carte moyenne de la main plutôt que sécuriser
    une carte forte contestée, qui peut disparaître au tour de l'adversaire.

    `tiebreak` (optionnel, `tiebreak(candidate_states, reference_state,
    observer) -> list[float]`, un score par candidat, positif si meilleur
    que la référence) : départage EN UN SEUL APPEL tous les candidats dont
    le gain exact est à moins de `tiebreak_margin` points du meilleur
    trouvé, comparés à ce meilleur (comparaison "en étoile", pas un
    tournoi séquentiel) -- PAS en concurrence avec le delta exact (qui
    reste le classement principal), seulement un second tour pour les cas
    où ce delta ne suffit déjà pas à distinguer les candidats de façon
    fiable. Voir `reference/value_policy.make_pairwise_gbm_tiebreak` et
    `reference/MODELS.md` ("Un modèle non linéaire discrimine mieux les
    paires serrées") : un modèle linéaire tourne au niveau du hasard sur
    cette tranche précise (delta exact < 3 pts) ; un Gradient Boosting
    entraîné sur les mêmes paires y gagne ~10 points de précision de
    signe. Le regroupement en un seul appel (plutôt qu'un appel par
    candidat) est délibéré : le coût mesuré vient d'un overhead fixe par
    appel sklearn, pas de la complexité de l'algorithme -- comparer 50
    candidats d'un coup ne coûte quasiment pas plus cher qu'en comparer 1
    (voir reference/MODELS.md), d'où un facteur ~10-50 de gain à regrouper.
    Non actif par défaut (opt-in), les décisions réelles et les rollouts
    existants ne changent pas de comportement tant que ce paramètre n'est
    pas fourni explicitement.
    """
    actions = game.legal_actions()
    if len(actions) == 1:
        return actions[0]
    if epsilon and rng.random() < epsilon:
        return rng.choice(actions)

    plays = [a for a in actions if a[0] not in ("draw", "skip_effect")]
    if not plays:
        return actions[0]
    if candidates and len(plays) > candidates:
        plays = rng.sample(plays, candidates)

    player = game.players[game.current]
    forest = player.forest
    base = forest.score()
    n_trees = forest.n_trees

    best, best_gain = None, -1e9
    gains = {}
    for a in plays:
        if a[0] == "tree":
            gain = forest.delta_tree(a[1]) + tree_bonus / (1 + n_trees)
            if tree_combo_bonus:
                gain += tree_combo_bonus.get(a[1], 0.0)
        elif a[0] == "cave_discard":
            # 1 pt par carte envoyée à la Grotte (Raton laveur) ; on ne
            # modélise pas ici la perte de valeur des cartes défaussées,
            # heuristique volontairement simple comme le reste du module.
            gain = a[1]
        else:
            _, did, tree_idx, pos = a
            forest.add_dweller(tree_idx, pos, did)
            gain = forest.score() - base
            forest.undo_dweller(tree_idx, pos, did)
        gains[a] = gain
        if gain > best_gain:
            best, best_gain = a, gain

    if tiebreak is not None:
        near = [a for a in plays if a != best and gains[a] >= best_gain - tiebreak_margin]
        if near:
            observer = game.current
            best_state = game.clone()
            best_state.apply(best)
            near_states = []
            for a in near:
                state = game.clone()
                state.apply(a)
                near_states.append(state)
            scores = tiebreak(near_states, best_state, observer)
            top = max(range(len(scores)), key=lambda i: scores[i])
            if scores[top] > 0:
                best = near[top]

    if (clearing_urgency and ("draw",) in actions
            and clearing_urgency > best_gain
            and any(_is_strong_card(c) for c in game.clearing)):
        return ("draw",)

    # Piocher est préférable si aucun coup ne rapporte et que la main respire.
    if best_gain <= 0 and len(player.hand) <= 7:
        return _fallback_action(game)
    return best


def greedy_policy(game, rng):
    return greedy_action(game, rng)


def beam_policy(game, rng, width=4, depth=2):
    """Recherche en faisceau sur les coups du joueur seul, adversaires figés."""
    actions = [a for a in game.legal_actions() if a[0] not in ("draw", "skip_effect")]
    if not actions:
        return _fallback_action(game)
    me = game.current
    beam = []
    for a in actions[:40]:
        probe = game.clone()
        probe.apply(a)
        beam.append((probe.players[me].forest.score(), a, probe))
    beam.sort(key=lambda x: -x[0])
    beam = beam[:width]

    for _ in range(depth - 1):
        nxt = []
        for _, first, state in beam:
            state.current = me  # on ignore les adversaires dans ce lookahead
            for a in [x for x in state.legal_actions() if x[0] not in ("draw", "skip_effect")][:20]:
                probe = state.clone()
                probe.apply(a)
                nxt.append((probe.players[me].forest.score(), first, probe))
        if not nxt:
            break
        nxt.sort(key=lambda x: -x[0])
        beam = nxt[:width]

    return beam[0][1]


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------


class Node:
    __slots__ = ("parent", "action", "player", "children", "visits", "value",
                 "value_sq", "untried")

    def __init__(self, parent, action, player):
        self.parent = parent
        self.action = action
        self.player = player      # joueur qui a joué `action`
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.value_sq = 0.0  # somme des carrés des retours, pour l'écart-type
        self.untried = None

    def uct_select(self, legal, c=C_PUCT, risk_k=0.0):
        """UCT restreint aux actions légales dans la déterminisation courante.

        Les compteurs de disponibilité (availability counts) de l'ISMCTS
        classique sont approximés par le total des visites des enfants légaux,
        ce qui suffit ici et évite un compteur par arête.

        `risk_k` : coefficient d'aversion au risque (session du 14/08, idée
        de Mehdi). À 0 (défaut), comportement inchangé : on exploite la
        moyenne des retours (`child.value / child.visits`). À risk_k > 0, on
        exploite `moyenne - risk_k * écart_type` : un nœud à forte variance
        est pénalisé même à moyenne égale, pour préférer des gains plus
        réguliers à des gains en espérance équivalente mais plus dispersés.
        L'écart-type est calculé sur les retours rétropropagés à ce nœud
        (E[X²] - E[X]², estimateur biaisé mais suffisant ici).
        """
        total = 1
        for a in legal:
            child = self.children.get(a)
            if child is not None:
                total += child.visits
        log_total = math.log(total)

        best, best_score = None, -1e18
        for a in legal:
            child = self.children.get(a)
            if child is None or child.visits == 0:
                return a, None
            mean = child.value / child.visits
            if risk_k:
                mean_sq = child.value_sq / child.visits
                std = math.sqrt(max(0.0, mean_sq - mean * mean))
                exploit = mean - risk_k * std
            else:
                exploit = mean
            explore = c * math.sqrt(log_total / child.visits)
            s = exploit + explore
            if s > best_score:
                best, best_score = a, s
        return best, self.children[best]


def determinize(game, observer, rng):
    """Rend au deck tout ce que `observer` ne peut pas voir, puis redistribue.

    L'observateur connaît sa propre main et toutes les forêts. Les mains
    adverses et l'ordre du deck sont réechantillonnés.
    """
    g = game.clone()
    sizes = []
    pool = []
    for i, p in enumerate(g.players):
        if i == observer:
            continue
        sizes.append((i, len(p.hand)))
        pool.extend(p.hand)
        p.hand = []

    # Les cartes Hiver restent dans le deck : elles ne peuvent pas se
    # retrouver en main, et leur position conditionne la fin de partie.
    winters = [c for c in g.deck if c[0] == WINTER]
    pool.extend(c for c in g.deck if c[0] != WINTER)
    rng.shuffle(pool)

    for i, n in sizes:
        g.players[i].hand = [pool.pop() for _ in range(min(n, len(pool)))]

    # On reinsère les cartes Hiver dans le dernier tiers restant, faute de
    # meilleure information : l'observateur sait combien ont été révélées,
    # pas où sont les autres.
    g.deck = insert_winter_cards(pool, rng, count=len(winters))
    g.rng = rng
    return g


def rollout(game, rng, max_moves=30, tree_combo_bonus=None):
    """Rollout tronqué puis évaluation par le score courant.

    Dérouler la partie jusqu'à l'hiver coûte ~70 coups ; tronquer à 30 et
    évaluer les forêts en l'état donne un signal presque aussi discriminant
    pour un tiers du coût. Le score d'une forêt est monotone croissant en
    cours de partie, ce qui rend l'évaluation tronquée peu biaisée. C'est
    l'équivalent bon marché d'une fonction de valeur, en attendant d'en
    apprendre une.

    `tree_combo_bonus` : voir `greedy_action`. Corrige, PENDANT le rollout
    seulement, le biais d'horizon du delta immédiat sur les arbres dont la
    valeur se révèle tard (ex. Sycomore, qui ne rapporte quasiment rien à
    la pose mais domine le classement d'espérance en fin de partie, voir
    docs/combo_guide.html) : sans lui, un rollout gouverné par le delta
    exact ne plante jamais un 2e/3e Sycomore, et sous-estime donc la vraie
    valeur des branches qui y mènent.
    """
    moves = 0
    while not game.over and moves < max_moves:
        actions = game.legal_actions()
        if len(actions) == 1:
            game.apply(actions[0])
        else:
            game.apply(greedy_action(game, rng, ROLLOUT_EPSILON,
                                     ROLLOUT_CANDIDATES,
                                     tree_combo_bonus=tree_combo_bonus))
        moves += 1
    return game.scores()


def reward_vector(scores):
    """Rang normalisé dans [0, 1], ex aequo partagés.

    Backpropager le score brut ferait apprendre à l'arbre la variance du
    tirage plutôt que la qualité des coups.
    """
    n = len(scores)
    if n == 1:
        return [1.0]
    out = []
    for s in scores:
        wins = sum(1 for o in scores if s > o)
        ties = sum(1 for o in scores if s == o) - 1
        out.append((wins + 0.5 * ties) / (n - 1))
    return out


class MCTS:
    """MCTS à information imparfaite, arbre persistant entre les tours.

    `leaf_eval`, si fourni, remplace le rollout tronqué par un appel
    direct à une fonction de valeur (ex. reference/value_policy.py) :
    signature `leaf_eval(state) -> list[float]` (un score estimé par
    joueur), appelée sur l'état atteint après sélection/expansion, sans
    dérouler la partie plus loin. Par défaut (`leaf_eval=None`), le
    comportement est inchangé (rollout aléatoire biaisé).

    `tiebreak` (optionnel, même signature que `greedy_action` :
    `tiebreak(candidate_states, reference_state, observer) -> list[float]`) :
    départage, UNE SEULE FOIS à la fin de `choose()` (pas à l'intérieur
    des simulations), les enfants de la racine dont le nombre de visites
    est à moins de `tiebreak_margin` (fraction du max, défaut 20%) du
    meilleur trouvé. Volontairement PAS branché dans `rollout()` : sur un
    échantillon réel, ~82% des tours à plusieurs candidats déclenchent un
    "presque à égalité" côté `greedy_action` -- brancher le modèle
    Gradient Boosting À CHAQUE étape simulée multiplierait le temps par
    décision par un facteur de plusieurs dizaines (des centaines
    d'itérations, chacune avec plusieurs coups de rollout). Un appel
    unique par décision réelle, lui, coûte ~2-3 ms de plus sur un budget
    de dizaines à centaines de ms -- négligeable. Voir reference/MODELS.md,
    "Comparateur pairwise dans greedy_action".
    """

    def __init__(self, observer, iterations=200, seed=None, c=C_PUCT,
                 rollout_depth=30, leaf_eval=None, risk_k=0.0,
                 tiebreak=None, tiebreak_margin=0.2):
        self.observer = observer
        self.iterations = iterations
        self.rollout_depth = rollout_depth
        self.c = c
        self.rng = random.Random(seed)
        self.root = Node(None, None, None)
        self.leaf_eval = leaf_eval
        self.risk_k = risk_k
        self.tiebreak = tiebreak
        self.tiebreak_margin = tiebreak_margin

    def advance(self, action):
        """Réutilise le sous-arbre correspondant au coup effectivement joué."""
        child = self.root.children.get(action)
        if child is None:
            child = Node(None, action, None)
        child.parent = None
        self.root = child

    def choose(self, game):
        root = self.root
        for _ in range(self.iterations):
            state = determinize(game, self.observer, self.rng)
            node = root
            # -- sélection
            while not state.over:
                legal = state.legal_actions()
                action, child = node.uct_select(legal, self.c, self.risk_k)
                if child is None:
                    # -- expansion
                    mover = state.current
                    state.apply(action)
                    node = node.children.setdefault(
                        action, Node(node, action, mover))
                    break
                mover = state.current
                state.apply(action)
                node = child
            # -- simulation (ou évaluation directe si leaf_eval fourni)
            if self.leaf_eval is not None:
                scores = self.leaf_eval(state)
            else:
                scores = rollout(state, self.rng, self.rollout_depth)
            rewards = reward_vector(scores)
            # -- rétropropagation
            while node is not None and node.parent is not None:
                node.visits += 1
                r = rewards[node.player]
                node.value += r
                node.value_sq += r * r
                node = node.parent
            root.visits += 1

        if not root.children:
            return _fallback_action(game)
        legal = set(game.legal_actions())
        candidates = [(c.visits, a) for a, c in root.children.items()
                      if a in legal]
        if not candidates:
            return _fallback_action(game)

        best_visits, best_action = max(candidates)

        if self.tiebreak is not None and best_visits > 0:
            threshold = best_visits * (1.0 - self.tiebreak_margin)
            near = [a for v, a in candidates if a != best_action and v >= threshold]
            if near:
                observer = game.current
                best_state = game.clone()
                best_state.apply(best_action)
                near_states = []
                for a in near:
                    state = game.clone()
                    state.apply(a)
                    near_states.append(state)
                scores = self.tiebreak(near_states, best_state, observer)
                top = max(range(len(scores)), key=lambda i: scores[i])
                if scores[top] > 0:
                    best_action = near[top]

        return best_action


def play_game(policies, n_players=2, seed=None, max_turns=1000):
    """Joue une partie complète. `policies[i](game, rng) -> action`."""
    game = Game(n_players=n_players, seed=seed)
    rng = random.Random(seed)
    turns = 0
    while not game.over and turns < max_turns:
        action = policies[game.current](game, rng)
        game.apply(action)
        turns += 1
    return game.scores(), turns
