"""
Simulateur de partie de Forêt Mixte (jeu de base).

Correction majeure par rapport à la version précédente du projet : le deck.
L'ancienne version piochait dans 250 cartes, en traitant chaque moitié
d'habitant comme une carte indépendante. La boîte contient en réalité :

    66 cartes Arbre
  + 48 cartes habitant Top/Bottom     (2 moitiés chacune -> 96 moitiés)
  + 44 cartes habitant Left/Right     (2 moitiés chacune ->  88 moitiés)
  ------------------------------------------------------------------
  = 158 cartes à piocher, portant 66 + 184 = 250 entités jouables

Piocher dans 250 cartes au lieu de 158 allongeait les parties d'environ 60 %,
décalait le déclenchement de l'hiver et rendait la grotte sans objet. Les
appariements réels des moitiés ne sont pas publiés ; ils sont ici tirés au
hasard à chaque partie (Top avec Bottom, Left avec Right), ce qui préserve la
bonne distribution marginale et la bonne taille de deck.

Clairière (confirmé par Mehdi) : les cartes défaussées en paiement rejoignent
`Game.clearing`, une zone commune face visible. À CHAQUE pioche du jeu — tour
normal ou effet de carte, sans distinction — le joueur peut prendre une carte
connue de la Clairière au lieu de piocher à l'aveugle dans le deck. Au-delà de
10 cartes, la Clairière est vidée (cartes perdues, pas remélangées).
`_draw_one` centralise ce choix pour que tous les points de pioche en
bénéficient automatiquement ; QUELLE carte prendre reste une heuristique
(`choose_draw_source`), pas une décision de l'arbre de recherche — l'exposer
multiplierait le facteur de branchement sur l'action la plus fréquente de la
partie, comme `choose_payment` déjà simplifié pour la même raison.

Planter un Arbre alimente AUSSI la Clairière depuis le deck (confirmé par
Mehdi, `_plant_tree_feeds_clearing`) : une carte du dessus du deck y est
placée en plus des cartes de paiement. C'est cette règle, pas juste le choix
de pioche ci-dessus, qui vide le deck et remplit la Clairière assez vite pour
borner la longueur d'une partie — sans elle, un joueur qui pioche toujours
dans une Clairière jamais vidée n'épuise quasiment plus le deck, ce qui
allonge les parties très au-delà des scores humains de référence (constaté
empiriquement avant d'ajouter cette règle).

Grotte : alimentée par l'effet du Raton laveur (CAVE_CHOICE_DWELLERS dans
engine.py, décision réelle) et par celui de l'Ours brun (CLEARING_TO_CAVE_
DWELLERS, inconditionnel : vide toute la Clairière dans sa Grotte). Il n'y a
pas de point gratuit par habitant posé.

Pose payante vs pose gratuite (confirmé par Mehdi) : une carte posée en
PAYANT son coût déclenche toujours normalement ses effets (pioche, rejeu,
bonus jumelles), que ce soit un tour normal ou la chaîne de la Taupe (Mole) —
c'est une action gratuite (pas de coût de TOUR), pas une pose gratuite. À
l'inverse, une carte posée VRAIMENT gratuitement (Blaireau/Lucane/Salamandre/
Moustique/Sapin blanc/Cerf élaphe) ne déclenche ni son effet ni son bonus.
`_pending_stack` (exposé en lecture via la propriété `pending_effect`, qui
retourne son sommet) est une vraie pile : une carte posée pendant la chaîne
de la Taupe peut elle-même ouvrir un nouvel effet en attente (Raton laveur,
une autre Taupe, un bonus de pose gratuite), qui s'empile et rend la main à
la chaîne une fois résolu. Un rejeu de tour gagné pendant la chaîne
(`pending_replay`) s'applique à sa fin, pas immédiatement.

Taille du deck selon le nombre de joueurs (confirmé par Mehdi,
`reduce_deck_for_player_count`) : 30 cartes retirées aléatoirement à 2
joueurs, 20 à 3, 10 à 4, aucune à 5 (deck complet). Retrait fait sur les 158
cartes physiques, avant la distribution des mains et avant l'insertion des
cartes Hiver (toujours exactement 3, jamais retirées).

Déclenchement de l'Hiver (confirmé par Mehdi, `insert_winter_cards`) : le
deck restant (après réduction et distribution des mains) est découpé en 3
tiers ; les 3 cartes Hiver vivent dans le DERNIER tiers pioché. La 1ère est
TOUJOURS la première carte piochée en entrant dans ce tiers (position
déterministe) ; les 2 autres sont mélangées aléatoirement dans le reste du
tiers. La 3e carte Hiver piochée (`winters_seen >= 3`) met fin à la partie.

Ce qui n'est PAS implémenté, et qui reste à faire pour un bot fiable :
  - bonus de paiement par couleur (Silver Fir, Douglas Fir, Oak, Badger,
    Fire Salamander, Stag Beetle)
  - moteurs de pioche des champignons (Chanterelle, Penny Bun, Parasol,
    Fly Agaric)
"""

import random

from engine import (
    CAVE_CHOICE_DWELLERS, CLEARING_TO_CAVE_DWELLERS, DRAW_FIXED,
    DRAW_IF_BONUS, DRAW_PER_COUNT, DWELLER_COST, DWELLER_NAME,
    DWELLER_PLAY_FREE_IF_BONUS, DWELLER_TYPE_IDS, FILTER_ANY_ANIMAL,
    FILTER_DWELLER, FILTER_TYPE, IS_ANIMAL, MUSHROOM_TRIGGER,
    MUSHROOM_TRIGGER_ANIMAL, MUSHROOM_TRIGGER_ANY_SYMBOL,
    MUSHROOM_TRIGGER_POSITION, N_DWELLERS, PLAY_CHAIN_DWELLERS, POS_ID,
    REPLAY_ALWAYS, REPLAY_IF_BONUS, SHARE_MAX, TREE_COPIES, TREE_COST,
    TREE_DRAW_FIXED, TREE_NAME, TREE_PLAY_FREE_IF_BONUS, TREE_REPLAY_ALWAYS,
    TREE_REPLAY_IF_BONUS, VARIANTS, Forest, score_players,
)

TREE, DWELLER, WINTER = 0, 1, 2

class IllegalAction(Exception):
    pass


HAND_LIMIT = 10
STARTING_HAND = 6


def build_deck(rng):
    """Construit les 158 cartes physiques + les 3 cartes Hiver.

    Retourne une liste où l'on pioche par la fin (pop()).
    """
    tops, bottoms, lefts, rights = [], [], [], []
    buckets = {POS_ID["Top"]: tops, POS_ID["Bottom"]: bottoms,
               POS_ID["Left"]: lefts, POS_ID["Right"]: rights}
    for did in range(N_DWELLERS):
        for tree_id, pos, copies in VARIANTS[did]:
            buckets[pos].extend([(did, tree_id, pos)] * copies)

    rng.shuffle(tops)
    rng.shuffle(bottoms)
    rng.shuffle(lefts)
    rng.shuffle(rights)

    cards = [(TREE, tree_id, None)
             for tree_id, n in enumerate(TREE_COPIES) for _ in range(n)]
    cards += [(DWELLER, a, b) for a, b in zip(tops, bottoms)]
    cards += [(DWELLER, a, b) for a, b in zip(lefts, rights)]
    rng.shuffle(cards)
    return cards  # 158 cartes ; les cartes Hiver sont insérées par Game


DECK_REDUCTION_BY_PLAYERS = {2: 30, 3: 20, 4: 10, 5: 0}


def reduce_deck_for_player_count(deck, n_players, rng):
    """Retire aléatoirement des cartes du deck selon le nombre de joueurs
    (confirmé par Mehdi) : 30 cartes en moins à 2 joueurs, 20 à 3, 10 à 4,
    aucune à 5 (on joue alors avec le deck complet). Les cartes retirées
    sortent définitivement de la partie ; ce retrait se fait AVANT la
    distribution des mains et avant l'insertion des cartes Hiver, sur les
    158 cartes physiques seulement (les cartes Hiver ne sont jamais
    retirées, il y en a toujours exactement 3).
    """
    n = DECK_REDUCTION_BY_PLAYERS.get(n_players, 0)
    if n <= 0:
        return deck
    remove = set(rng.sample(range(len(deck)), n))
    return [c for i, c in enumerate(deck) if i not in remove]


def insert_winter_cards(deck, rng, count=3):
    """Découpe le deck restant (après réduction par nombre de joueurs et
    distribution des mains) en 3 tiers ; les cartes Hiver vivent dans le
    DERNIER tiers pioché. On pioche par la fin de la liste, donc le dernier
    tiers pioché est le début de la liste (`deck[:third]`).

    Confirmé par Mehdi : la 1ère carte Hiver est TOUJOURS la première
    piochée en entrant dans ce tiers -- position déterministe, pas
    aléatoire. Les `count - 1` cartes Hiver restantes sont mélangées
    aléatoirement dans le reste du tiers. La 3e carte Hiver piochée met fin
    à la partie (`Game._draw_one`, `Game._plant_tree_feeds_clearing`).

    L'insertion se fait après la distribution des mains d'ouverture, ce qui
    est équivalent (les 6 premières cartes de chaque joueur ne peuvent pas
    venir du dernier tiers) et évite qu'un mulligan ne remette une carte
    Hiver en circulation.
    """
    third = len(deck) // 3
    tail = deck[:third]
    for _ in range(count - 1):
        tail.insert(rng.randrange(len(tail) + 1), (WINTER, None, None))
    tail.append((WINTER, None, None))  # 1ère piochée : position déterministe
    return tail + deck[third:]


def card_cost(card, half_index):
    kind, a, b = card
    if kind == TREE:
        return TREE_COST[a]
    half = a if half_index == 0 else b
    return DWELLER_COST[half[0]]


def card_symbol(card, half_index):
    """Symbole d'arbre imprimé sur la moitié jouée (None pour un Arbre)."""
    kind, a, b = card
    if kind != DWELLER:
        return None
    half = a if half_index == 0 else b
    return half[1]


def card_min_cost(card):
    """Coût le plus bas entre les deux moitiés jouables (un Arbre n'a qu'un
    coût). Sert à choisir la carte la plus "flexible" dans la Clairière.
    """
    if card[0] == TREE:
        return card_cost(card, 0)
    return min(card_cost(card, 0), card_cost(card, 1))


def choose_draw_source(clearing):
    """Choix pioche Clairière vs pioche aveugle (deck) par défaut, pour tous
    les points de pioche déclenchés par un EFFET de carte (bonus jumelles,
    Fougère arborescente, Ours brun, etc. -- voir `_draw_one`).

    Heuristique, pas une décision de recherche : exposer CES pioches-là à
    l'arbre multiplierait le facteur de branchement pour un gain incertain
    (elles ne sont pas l'action la plus fréquente de la partie, contrairement
    à la pioche de tour normal -- voir `AUTO`/`Game.legal_actions`, qui
    EXPOSE ce choix-ci). Une carte connue de la Clairière est presque
    toujours au moins aussi utile qu'une carte aveugle inconnue (on peut
    toujours la garder sans la jouer), donc on prend systématiquement la
    moins chère de la Clairière quand elle est non vide ; pioche aveugle
    sinon. Simplification assumée, comme `choose_payment` et la sélection
    des cartes envoyées à la Grotte.

    Retourne l'indice dans `clearing` à prendre, ou None pour piocher dans
    le deck.
    """
    if not clearing:
        return None
    return min(range(len(clearing)), key=lambda i: card_min_cost(clearing[i]))


AUTO = object()  # sentinel : _draw_one doit decider via choose_draw_source


class Player:
    __slots__ = ("hand", "forest")

    def __init__(self):
        self.hand = []
        self.forest = Forest()

    def copy(self):
        p = Player.__new__(Player)
        p.hand = self.hand[:]
        p.forest = self.forest.copy()
        return p


class Game:
    """État complet d'une partie. `clone()` est conçu pour être appelé des
    milliers de fois par la recherche, il évite toute copie profonde inutile.
    """

    __slots__ = ("players", "deck", "clearing", "current", "winters_seen",
                 "over", "rng", "last_bonus_paid",
                 "_pending_stack", "pending_replay")

    def __init__(self, n_players=2, seed=None):
        self.rng = random.Random(seed)
        self.deck = build_deck(self.rng)
        self.deck = reduce_deck_for_player_count(self.deck, n_players, self.rng)
        self.players = [Player() for _ in range(n_players)]
        self.clearing = []
        self.current = 0
        self.winters_seen = 0
        self.over = False
        self.last_bonus_paid = False
        self._pending_stack = []
        self.pending_replay = False
        for p in self.players:
            self._deal_opening_hand(p)
        self.deck = insert_winter_cards(self.deck, self.rng)

    @property
    def pending_effect(self):
        """Sommet de la pile d'effets en attente (ou None). Confirmé par
        Mehdi : une carte posée pendant la chaîne de la Taupe est payée
        normalement, donc génère ses effets normalement -- y compris,
        potentiellement, un NOUVEL effet en attente (Raton laveur, une autre
        Taupe, Blaireau/Lucane/Salamandre/Sapin blanc/Cerf élaphe si bonus).
        `_pending_stack` est une vraie pile pour supporter cette imbrication ;
        cette propriété n'expose que le sommet, pour que legal_actions() et
        le code externe (search.py, reference/) n'aient rien à changer.
        """
        return self._pending_stack[-1] if self._pending_stack else None

    def _deal_opening_hand(self, player):
        for _ in range(STARTING_HAND):
            player.hand.append(self.deck.pop())
        # Mulligan : autorisé une fois si la main d'ouverture n'a aucun arbre.
        if not any(c[0] == TREE for c in player.hand):
            self.deck.extend(player.hand)
            self.rng.shuffle(self.deck)
            player.hand = [self.deck.pop() for _ in range(STARTING_HAND)]

    def clone(self):
        g = Game.__new__(Game)
        g.rng = self.rng
        g.deck = self.deck[:]
        g.players = [p.copy() for p in self.players]
        g.clearing = self.clearing[:]
        g.current = self.current
        g.winters_seen = self.winters_seen
        g.over = self.over
        g.last_bonus_paid = self.last_bonus_paid
        g._pending_stack = self._pending_stack[:]
        g.pending_replay = self.pending_replay
        return g

    # -- pioche ------------------------------------------------------------

    def _draw_one(self, player, source=AUTO):
        """`source` : None = pioche aveugle forcée, un entier = index dans
        `self.clearing` à prendre, `AUTO` (défaut) = laisser
        `choose_draw_source` décider. Seule la pioche de tour normal
        (`Game.legal_actions`, action `("draw", source)`) passe un `source`
        explicite -- une vraie décision de recherche ; tous les autres
        points de pioche (effets de carte) restent en `AUTO`, voir
        `choose_draw_source`.
        """
        idx = choose_draw_source(self.clearing) if source is AUTO else source
        if idx is not None:
            player.hand.append(self.clearing.pop(idx))
            return
        if not self.deck:
            self.over = True
            return
        card = self.deck.pop()
        if card[0] == WINTER:
            self.winters_seen += 1
            if self.winters_seen >= 3:
                self.over = True
            return
        player.hand.append(card)

    def _add_to_clearing(self, card):
        """Ajoute une carte défaussée en paiement à la Clairière (zone
        commune face visible). Vidage à 10 confirmé par Mehdi : au-delà, les
        cartes sont perdues (retirées définitivement de la partie), pas
        remélangées dans le deck.
        """
        self.clearing.append(card)
        if len(self.clearing) >= 10:
            self.clearing.clear()

    def _plant_tree_feeds_clearing(self):
        """Planter un Arbre alimente aussi la Clairière depuis le DECK
        (confirmé par Mehdi), en plus des éventuelles cartes de paiement :
        c'est ce qui vide le deck et remplit la Clairière assez vite pour
        borner la longueur d'une partie (sans cette règle, la Clairière ne
        se videait quasiment jamais, ce qui allongeait les parties très
        au-delà des scores humains observés). Une carte Hiver révélée ainsi
        compte comme rencontrée, exactement comme une pioche aveugle, et ne
        rejoint pas la Clairière.
        """
        if not self.deck:
            self.over = True
            return
        card = self.deck.pop()
        if card[0] == WINTER:
            self.winters_seen += 1
            if self.winters_seen >= 3:
                self.over = True
            return
        self._add_to_clearing(card)

    # -- actions -----------------------------------------------------------

    def legal_actions(self):
        """Actions du joueur courant, identifiées de façon invariante.

        Une action est ("draw", source), ("tree", tree_id),
        ("dweller", dweller_id, index_d_arbre, position), ou, si un effet
        "jouer gratuitement depuis la main" est en attente (voir
        `pending_effect`, brique 2b) : ("free_dweller", dweller_id,
        index_d_arbre, position) ou ("skip_effect",).

        `source` (pioche de tour normal, exposée à la recherche -- confirmé
        par Mehdi que c'est la décision, pas juste la mécanique, qui doit
        être jouable) : None pioche le deck à l'aveugle, un entier prend la
        carte à cet index dans `self.clearing`. Une carte par option pour
        rester alignée avec la carte physique réellement prise (pas un choix
        "petit/grand prix" agrégé) ; la 2e pioche éventuelle (main sous
        HAND_LIMIT après la 1ère) reste `AUTO` -- voir `Game._draw_one`,
        simplification assumée comme le paiement.

        Point important pour la recherche : l'action ne référence PAS l'indice
        de la carte en main. Cet indice se décale à chaque pioche et à chaque
        défausse, donc une action indexée par la main désignerait des coups
        différents d'une déterminisation à l'autre et l'arbre MCTS mélangerait
        des statistiques sans rapport. Identifier l'action par la carte jouée
        règle le problème et fusionne au passage les doublons de main, ce qui
        réduit nettement le facteur de branchement.
        """
        player = self.players[self.current]
        hand = player.hand
        forest = player.forest

        if self.pending_effect is not None:
            kind = self.pending_effect[0]
            if kind == "cave_choice":
                return [("cave_discard", n) for n in range(len(hand) + 1)]
            if kind == "play_chain":
                return [("skip_effect",)] + self._payable_actions(hand, forest)
            filter_kind, filter_arg, _remaining = self.pending_effect
            actions = [("skip_effect",)]
            seen = set()
            for card in hand:
                if card[0] != DWELLER:
                    continue
                for half in (card[1], card[2]):
                    did, _tree_id, pos = half
                    if not self._matches_filter(did, filter_kind, filter_arg):
                        continue
                    for tree_idx in self._slots_for(forest, pos, did):
                        key = ("free_dweller", did, tree_idx, pos)
                        if key not in seen:
                            seen.add(key)
                            actions.append(key)
            return actions

        draws = [("draw", None)] + [("draw", i) for i in range(len(self.clearing))]
        return draws + self._payable_actions(hand, forest)

    @staticmethod
    def _payable_actions(hand, forest):
        """Poses payantes normales (arbre ou habitant) affordables avec la
        main actuelle. Partagé entre le tour normal et la chaîne de la
        Taupe (`pending_effect` de type "play_chain").
        """
        actions = []
        seen = set()
        budget = len(hand) - 1  # la carte jouée ne peut pas se payer elle-même

        for card in hand:
            kind, a, b = card
            if kind == TREE:
                if TREE_COST[a] <= budget:
                    key = ("tree", a)
                    if key not in seen:
                        seen.add(key)
                        actions.append(key)
                continue
            if kind != DWELLER:
                continue
            for half in (a, b):
                did, _tree_id, pos = half
                if DWELLER_COST[did] > budget:
                    continue
                for tree_idx in Game._slots_for(forest, pos, did):
                    key = ("dweller", did, tree_idx, pos)
                    if key not in seen:
                        seen.add(key)
                        actions.append(key)
        return actions

    @staticmethod
    def _matches_filter(did, filter_kind, filter_arg):
        if filter_kind == FILTER_ANY_ANIMAL:
            return IS_ANIMAL[did]
        if filter_kind == FILTER_TYPE:
            return filter_arg in DWELLER_TYPE_IDS[did]
        if filter_kind == FILTER_DWELLER:
            return did == filter_arg
        return False

    @staticmethod
    def _slots_for(forest, pos, dweller_id):
        """Emplacements légaux : seul le côté (pos) compte, pas l'espèce.

        Règle officielle (rulebook FR) : un habitant se pose sur n'importe
        quel arbre ayant un slot vide du bon côté.
        """
        cap = SHARE_MAX[dweller_id]
        for idx in range(len(forest.species)):
            slot = forest.slots[idx][pos]
            if not slot:
                yield idx
            elif cap and slot[0][0] == dweller_id and (cap < 0 or len(slot) < cap):
                yield idx

    def _find_card(self, hand, action):
        """Retrouve la carte et la moitié qui réalisent l'action.

        Le matching se fait sur (dweller_id, position) : l'espèce ne
        restreint pas le placement. S'il y a plusieurs candidates (variantes
        différentes en main), on prend la première ; laquelle sacrifier est
        une décision de jeu réelle laissée hors de l'arbre pour l'instant,
        comme le paiement. Retourne aussi le symbole imprimé de la moitié
        choisie (nécessaire pour ROE_DEER, voir Forest.add_dweller).
        """
        if action[0] == "tree":
            target = (TREE, action[1], None)
            for i, card in enumerate(hand):
                if card == target:
                    return i, 0, None
            return None
        _, did, tree_idx, pos = action
        for i, card in enumerate(hand):
            if card[0] != DWELLER:
                continue
            for half_index, half in ((0, card[1]), (1, card[2])):
                if half[0] == did and half[2] == pos:
                    return i, half_index, half[1]
        return None

    def _resolve_mushroom_triggers(self, player, did, pos):
        """Champignons à effet permanent (brique 3) : pioche déclenchée par
        CHAQUE habitant posé par ce joueur qui remplit leur condition, tant
        que le champignon est en jeu. Voir engine.py (MUSHROOM_TRIGGER) pour
        les conditions et les hypothèses documentées.
        """
        is_animal = IS_ANIMAL[did]
        for mushroom_did, (kind, arg) in MUSHROOM_TRIGGER.items():
            n = player.forest.dweller_count[mushroom_did]
            if n <= 0:
                continue
            if kind == MUSHROOM_TRIGGER_ANIMAL and not is_animal:
                continue
            if kind == MUSHROOM_TRIGGER_POSITION and pos != arg:
                continue
            # MUSHROOM_TRIGGER_ANY_SYMBOL : toujours vrai pour un dweller.
            for _ in range(n):
                self._draw_one(player)

    def apply(self, action, payment=None):
        player = self.players[self.current]

        if self._pending_stack:
            self._apply_pending(player, action)
            return self

        if action[0] == "draw":
            self._draw_one(player, action[1])
            if len(player.hand) < HAND_LIMIT and not self.over:
                self._draw_one(player)
        else:
            self._play_card(player, action, payment)
        self._end_action()
        return self

    def _end_action(self):
        """Décide si le tour passe au joueur suivant, après résolution (ou
        fermeture) d'un effet. Ne fait rien tant qu'il reste un effet sur la
        pile (`_pending_stack`) -- la main revient alors au niveau en
        dessous (ex. la chaîne de la Taupe, après la résolution d'un Raton
        laveur posé pendant la chaîne). Une fois la pile vide, un rejeu de
        tour accumulé pendant l'action (Geai des chênes, Loup/Sapin Douglas/
        Chêne si bonus) est consommé avant de faire passer le tour --
        confirmé par Mehdi : un rejeu gagné PENDANT la chaîne de la Taupe
        s'ajoute à la fin, il ne se perd pas juste parce que la chaîne
        continuait au moment où il a été gagné.
        """
        if self._pending_stack:
            return
        if self.pending_replay:
            self.pending_replay = False
            return
        self.current = (self.current + 1) % len(self.players)

    def _play_card(self, player, action, payment=None):
        """Pose payante d'une carte (arbre ou habitant) : paiement, ajout en
        forêt, effets de pose. Partagé entre le tour normal (`apply`) et la
        chaîne de la Taupe (`_apply_pending`, "play_chain") : confirmé par
        Mehdi, une carte posée en payant son coût déclenche normalement ses
        effets (pioche, rejeu, bonus jumelles...), chaîne ou pas -- c'est une
        action gratuite (pas de coût de TOUR), pas une pose gratuite. À
        l'inverse, une vraie pose gratuite (Blaireau/Lucane/Salamandre/
        Moustique/Sapin blanc/Cerf élaphe, voir `_apply_pending` ci-dessous,
        cas "free_dweller") ne déclenche ni son effet ni son bonus, câblée
        séparément et volontairement plus simple.
        """
        found = self._find_card(player.hand, action)
        if found is None:
            raise IllegalAction(f"action impossible dans cet état : {action}")
        hand_index, half_index, symbol = found
        card = player.hand.pop(hand_index)
        cost = card_cost(card, half_index)
        if action[0] == "tree":
            # Le symbole "pertinent" pour le bonus d'un arbre est l'espèce
            # elle-même (ex. Sapin blanc payé avec une carte de symbole
            # Sapin blanc).
            bonus_symbol = action[1]
        else:
            bonus_symbol = symbol
        if payment is None:
            payment = choose_payment(player.hand, cost, preferred_symbol=bonus_symbol)
        self.last_bonus_paid = False
        if bonus_symbol is not None and payment:
            self.last_bonus_paid = any(
                card_symbol(player.hand[j], 0) == bonus_symbol
                or card_symbol(player.hand[j], 1) == bonus_symbol
                for j in payment
            )
        for j in sorted(payment, reverse=True):
            self._add_to_clearing(player.hand.pop(j))

        replay = False
        if action[0] == "tree":
            tree_id = action[1]
            player.forest.add_tree(tree_id)
            self._plant_tree_feeds_clearing()
            for _ in range(TREE_DRAW_FIXED.get(tree_id, 0)):
                self._draw_one(player)
            if tree_id in TREE_REPLAY_ALWAYS:
                replay = True
            elif self.last_bonus_paid and tree_id in TREE_REPLAY_IF_BONUS:
                replay = True
            if self.last_bonus_paid and tree_id in TREE_PLAY_FREE_IF_BONUS:
                filter_kind, filter_arg, remaining = TREE_PLAY_FREE_IF_BONUS[tree_id]
                self._pending_stack.append((filter_kind, filter_arg, remaining))
        else:
            _, did, tree_idx, pos = action
            player.forest.add_dweller(tree_idx, pos, did, symbol)
            self._resolve_mushroom_triggers(player, did, pos)
            replay = self._resolve_dweller_effect(player, did)
            if did in CAVE_CHOICE_DWELLERS:
                self._pending_stack.append(("cave_choice", None, None))
            elif did in PLAY_CHAIN_DWELLERS:
                self._pending_stack.append(("play_chain", None, None))
            elif self.last_bonus_paid and did in DWELLER_PLAY_FREE_IF_BONUS:
                filter_kind, filter_arg, remaining = DWELLER_PLAY_FREE_IF_BONUS[did]
                self._pending_stack.append((filter_kind, filter_arg, remaining))

        self.pending_replay = self.pending_replay or replay

    def _apply_pending(self, player, action):
        """Résout une action liée à l'effet en tête de `_pending_stack`
        (sommet de la pile, voir `pending_effect`). brique 2b : jouer
        gratuitement une carte depuis la main ; brique 2c : envoyer des
        cartes à la Grotte ; Taupe : chaîne de poses payantes, potentiellement
        imbriquée (une carte posée pendant la chaîne peut elle-même ouvrir un
        nouvel effet, empilé par-dessus -- voir `_play_card`).
        """
        kind = self.pending_effect[0]

        if kind == "cave_choice":
            if action[0] != "cave_discard":
                raise IllegalAction(f"action impossible pendant l'effet Grotte : {action}")
            n = action[1]
            # Comme choose_payment : heuristique de sélection des cartes
            # envoyées à la Grotte (les plus chères d'abord, on garde les
            # cartes bon marché et polyvalentes en main). Le NOMBRE reste
            # une vraie décision de l'arbre ; LESQUELLES ne l'est pas ici.
            ranked = sorted(
                range(len(player.hand)),
                key=lambda i: -min(card_cost(player.hand[i], 0), card_cost(player.hand[i], 1))
                if player.hand[i][0] == DWELLER else -TREE_COST[player.hand[i][1]],
            )
            for i in sorted(ranked[:n], reverse=True):
                player.hand.pop(i)
            player.forest.cave += n
            for _ in range(n):
                self._draw_one(player)
            self._pending_stack.pop()
            self._end_action()
            return

        if kind == "play_chain":
            if action[0] == "skip_effect":
                self._pending_stack.pop()
                self._end_action()
                return
            if action[0] not in ("tree", "dweller"):
                raise IllegalAction(f"action impossible pendant la chaîne Taupe : {action}")
            self._play_card(player, action)
            # La chaîne elle-même reste sur la pile (elle ne se termine que
            # par skip_effect ou l'absence de coup finançable, détectée à la
            # prochaine legal_actions()). Si _play_card a empilé un nouvel
            # effet (ex. Raton laveur posé pendant la chaîne), il devient le
            # nouveau sommet et sera résolu avant de revenir à la chaîne ;
            # _end_action() ne fait rien tant que la pile n'est pas vide.
            self._end_action()
            return

        if action[0] == "skip_effect":
            self._pending_stack.pop()
            self._end_action()
            return
        if action[0] != "free_dweller":
            raise IllegalAction(f"action impossible pendant un effet en attente : {action}")
        _, did, tree_idx, pos = action
        filter_kind, filter_arg, remaining = self.pending_effect
        if not self._matches_filter(did, filter_kind, filter_arg):
            raise IllegalAction(f"cette carte ne correspond pas à l'effet en attente : {action}")
        for i, card in enumerate(player.hand):
            if card[0] != DWELLER:
                continue
            for half_index, half in ((0, card[1]), (1, card[2])):
                if half[0] == did and half[2] == pos:
                    symbol = half[1]
                    player.hand.pop(i)
                    player.forest.add_dweller(tree_idx, pos, did, symbol)
                    self._resolve_mushroom_triggers(player, did, pos)
                    if remaining is not None and remaining <= 1:
                        self._pending_stack.pop()
                    else:
                        new_remaining = None if remaining is None else remaining - 1
                        self._pending_stack[-1] = (filter_kind, filter_arg, new_remaining)
                    self._end_action()
                    return
        raise IllegalAction(f"carte introuvable pour l'effet en attente : {action}")

    def _resolve_dweller_effect(self, player, did):
        """Effets de pose sans sous-choix (brique 2, batch 1) : pioche et
        rejeu de tour. Retourne True si le joueur courant doit rejouer.
        Voir engine.py pour les tables et reference/cartes_effets.md pour
        la source des règles.
        """
        if did in CLEARING_TO_CAVE_DWELLERS:
            # Ours brun : effet inconditionnel, avant le tirage/rejeu bonus
            # jumelles éventuel (tables génériques ci-dessous).
            player.forest.cave += len(self.clearing)
            self.clearing.clear()
        for _ in range(DRAW_FIXED.get(did, 0)):
            self._draw_one(player)
        if self.last_bonus_paid:
            for _ in range(DRAW_IF_BONUS.get(did, 0)):
                self._draw_one(player)
        per_count = DRAW_PER_COUNT.get(did)
        if per_count is not None:
            kind, ref_id = per_count
            n = (player.forest.dweller_count[ref_id] if kind == "dweller"
                 else player.forest.type_count[ref_id])
            for _ in range(n):
                self._draw_one(player)
        if did in REPLAY_ALWAYS:
            return True
        if did in REPLAY_IF_BONUS and self.last_bonus_paid:
            return True
        return False

    def scores(self):
        return score_players([p.forest for p in self.players])


def choose_payment(hand, cost, preferred_symbol=None):
    """Choix des cartes défaussées pour payer.

    Heuristique, pas une décision de recherche : on garde les cartes bon
    marché et polyvalentes, on défausse celles dont la meilleure moitié coûte
    le plus cher. C'est une simplification assumée. Rendre le paiement
    lui-même une décision de l'arbre multiplierait l'espace d'actions par le
    nombre de combinaisons de défausse, et c'est le premier endroit où gagner
    en qualité de jeu une fois le reste stabilisé.

    `preferred_symbol` : symbole imprimé sur la carte qu'on est en train de
    poser. Beaucoup de dwellers ont un "bonus jumelles" qui se déclenche si
    AU MOINS UNE carte de la défausse porte ce même symbole (confirmé par
    Mehdi : ex. le Loup porte le symbole Sapin blanc, payer avec au moins
    une carte Sapin blanc déclenche son bonus). Quand un symbole préféré est
    fourni, on priorise la défausse des cartes qui le portent, pour
    maximiser les chances de déclencher le bonus sans changer le coût payé.
    """
    if cost <= 0:
        return []

    def best_cost(i):
        c = hand[i]
        if c[0] == DWELLER:
            return min(card_cost(c, 0), card_cost(c, 1))
        return TREE_COST[c[1]]

    def matches_preferred(i):
        if preferred_symbol is None:
            return False
        c = hand[i]
        if c[0] != DWELLER:
            return False
        return card_symbol(c, 0) == preferred_symbol or card_symbol(c, 1) == preferred_symbol

    ranked = sorted(
        range(len(hand)),
        key=lambda i: (0 if matches_preferred(i) else 1, -best_cost(i)),
    )
    return ranked[:cost]


def card_label(card):
    kind, a, b = card
    if kind == TREE:
        return f"Arbre:{TREE_NAME[a]}"
    if kind == WINTER:
        return "Hiver"
    return f"{DWELLER_NAME[a[0]]}/{DWELLER_NAME[b[0]]}"
