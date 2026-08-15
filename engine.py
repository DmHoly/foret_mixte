"""
Moteur de score optimisé de Forêt Mixte (jeu de base).

Principe : toutes les quantités dont dépendent les règles sont maintenues
incrémentalement à la pose d'une carte. `score()` ne parcourt donc jamais la
forêt : il combine une trentaine de compteurs et une boucle sur les 8 espèces
d'arbres. Le coût du scoring devient indépendant du nombre d'habitants posés,
ce qui est la propriété qui compte pour des rollouts MCTS.

L'oracle de correction est `scoring_ref.py` (transcription littérale du moteur
TypeScript de référence). `tests/test_rules.py` compare les deux sur des
forêts aléatoires et sur les cas unitaires du dépôt d'origine.

Correctifs par rapport à la version précédente du projet :
  - ROE_DEER comptait les arbres de même espèce ; la règle compte toutes les
    cartes portant le même symbole d'arbre, habitants compris.
  - COMMON_TOAD marquait 5 points par arbre apparié ; chaque crapaud marque
    5 points, donc 10 pour une paire.
  - VIOLET_CARPENTER_BEE n'était pas implémentée : elle fait compter son arbre
    porteur une fois de plus (MOSS, majorité d'arbres).
  - Les positions autorisées de 7 habitants étaient trop permissives.
  - La grotte était incrémentée à chaque habitant posé, y compris quand la
    moitié inutilisée n'existe pas.
"""

from cards import DWELLERS, POSITIONS, SLOT_SHARING, TREES

# ---------------------------------------------------------------------------
# Tables statiques (construites une fois à l'import)
# ---------------------------------------------------------------------------

N_TREES = len(TREES)
N_DWELLERS = len(DWELLERS)

TREE_ID = {t.name: i for i, t in enumerate(TREES)}
TREE_NAME = [t.name for t in TREES]
TREE_COST = [t.cost for t in TREES]
TREE_COPIES = [t.copies for t in TREES]

DWELLER_ID = {d.name: i for i, d in enumerate(DWELLERS)}
DWELLER_NAME = [d.name for d in DWELLERS]
DWELLER_COST = [d.cost for d in DWELLERS]

POS_ID = {p: i for i, p in enumerate(POSITIONS)}
TOP, BOTTOM, LEFT, RIGHT = (POS_ID[p] for p in ("Top", "Bottom", "Left", "Right"))
OPPOSITE = {LEFT: RIGHT, RIGHT: LEFT}

ALL_TYPES = ("Amphibian", "Bat", "Bird", "Butterfly", "ClovenhoofedAnimal",
             "Deer", "Insect", "Mushroom", "PawedAnimal", "Plant")
TYPE_ID = {t: i for i, t in enumerate(ALL_TYPES)}
N_TYPES = len(ALL_TYPES)

# type_of[dweller_id] -> tuple d'indices de type (les habitants ont 1 ou 2 types)
DWELLER_TYPE_IDS = tuple(
    tuple(TYPE_ID[t] for t in d.types) for d in DWELLERS
)

# placements[dweller_id] -> frozenset de (tree_id, pos_id) autorisés
PLACEMENTS = tuple(
    frozenset((TREE_ID[v.tree], POS_ID[v.position]) for v in d.variants)
    for d in DWELLERS
)
# Positions valides pour un habitant, indépendamment de l'espèce d'arbre :
# propriété physique fixe de la carte (moitié Top/Bottom ou Left/Right),
# à ne pas confondre avec l'espèce imprimée qui ne restreint pas le
# placement (voir Forest.can_place / legal_positions).
VALID_POS = tuple(
    frozenset(POS_ID[v.position] for v in d.variants)
    for d in DWELLERS
)
# copies par variante, pour construire le deck
VARIANTS = tuple(
    tuple((TREE_ID[v.tree], POS_ID[v.position], v.copies) for v in d.variants)
    for d in DWELLERS
)

SHARE_MAX = [0] * N_DWELLERS  # 0 = pas de partage, -1 = illimité, n = max n
for name, cap in SLOT_SHARING.items():
    SHARE_MAX[DWELLER_ID[name]] = -1 if cap is None else cap

# Identifiants utilisés dans les règles
D_BEECH_MARTEN = DWELLER_ID["BEECH_MARTEN"]
D_BLACKBERRIES = DWELLER_ID["BLACKBERRIES"]
D_BROWN_BEAR = DWELLER_ID["BROWN_BEAR"]
D_BULLFINCH = DWELLER_ID["BULLFINCH"]
D_CHAFFINCH = DWELLER_ID["CHAFFINCH"]
D_COMMON_TOAD = DWELLER_ID["COMMON_TOAD"]
D_EURASIAN_JAY = DWELLER_ID["EURASIAN_JAY"]
D_EUROPEAN_BADGER = DWELLER_ID["EUROPEAN_BADGER"]
D_FAT_DORMOUSE = DWELLER_ID["EUROPEAN_FAT_DORMOUSE"]
D_HARE = DWELLER_ID["EUROPEAN_HARE"]
D_FALLOW_DEER = DWELLER_ID["FALLOW_DEER"]
D_SALAMANDER = DWELLER_ID["FIRE_SALAMANDER"]
D_FIREFLIES = DWELLER_ID["FIREFLIES"]
D_GNAT = DWELLER_ID["GNAT"]
D_GOSHAWK = DWELLER_ID["GOSHAWK"]
D_WOODPECKER = DWELLER_ID["GREAT_SPOTTED_WOODPECKER"]
D_HEDGEHOG = DWELLER_ID["HEDGEHOG"]
D_LYNX = DWELLER_ID["LYNX"]
D_MOSS = DWELLER_ID["MOSS"]
D_POND_TURTLE = DWELLER_ID["POND_TURTLE"]
D_RED_DEER = DWELLER_ID["RED_DEER"]
D_RED_FOX = DWELLER_ID["RED_FOX"]
D_RED_SQUIRREL = DWELLER_ID["RED_SQUIRREL"]
D_ROE_DEER = DWELLER_ID["ROE_DEER"]
D_SQUEAKER = DWELLER_ID["SQUEAKER"]
D_STAG_BEETLE = DWELLER_ID["STAG_BEETLE"]
D_TAWNY_OWL = DWELLER_ID["TAWNY_OWL"]
D_TREE_FERNS = DWELLER_ID["TREE_FERNS"]
D_TREE_FROG = DWELLER_ID["TREE_FROG"]
D_BEE = DWELLER_ID["VIOLET_CARPENTER_BEE"]
D_WILD_BOAR = DWELLER_ID["WILD_BOAR"]
D_STRAWBERRIES = DWELLER_ID["WILD_STRAWBERRIES"]
D_WOLF = DWELLER_ID["WOLF"]
D_WOOD_ANT = DWELLER_ID["WOOD_ANT"]

T_BEECH = TREE_ID["BEECH"]
T_BIRCH = TREE_ID["BIRCH"]
T_DOUGLAS_FIR = TREE_ID["DOUGLAS_FIR"]
T_HORSE_CHESTNUT = TREE_ID["HORSE_CHESTNUT"]
T_LINDEN = TREE_ID["LINDEN"]
T_OAK = TREE_ID["OAK"]
T_SILVER_FIR = TREE_ID["SILVER_FIR"]
T_SYCAMORE = TREE_ID["SYCAMORE"]

TY_AMPHIBIAN = TYPE_ID["Amphibian"]
TY_BAT = TYPE_ID["Bat"]
TY_BIRD = TYPE_ID["Bird"]
TY_BUTTERFLY = TYPE_ID["Butterfly"]
TY_CLOVEN = TYPE_ID["ClovenhoofedAnimal"]
TY_DEER = TYPE_ID["Deer"]
TY_INSECT = TYPE_ID["Insect"]
TY_MUSHROOM = TYPE_ID["Mushroom"]
TY_PAWED = TYPE_ID["PawedAnimal"]
TY_PLANT = TYPE_ID["Plant"]

# "Animal" au sens du jeu = tout habitant dont aucun type n'est Plante ni
# Champignon (ces deux catégories sont les seules non-animales du jeu).
IS_ANIMAL = tuple(
    TY_PLANT not in types and TY_MUSHROOM not in types
    for types in DWELLER_TYPE_IDS
)

# ---------------------------------------------------------------------------
# Effets de pose (brique 2 du refactoring, voir reference/REFACTOR_PLAN.md
# et reference/cartes_effets.md pour la source des règles).
#
# Batch "sans sous-choix" uniquement : pioche (simple, ou conditionnée au
# bonus jumelles, ou proportionnelle à un compteur de forêt) et rejeu de
# tour. Les effets qui demandent un sous-choix du joueur (jouer gratuitement
# une carte précise depuis la main) sont un batch séparé (brique 2b), pas
# encore implémentés : ils exigent un noeud de décision imbriqué.
#
# DRAW_FIXED[dweller_id]      -> nombre de cartes à piocher, sans condition.
# DRAW_IF_BONUS[dweller_id]   -> nombre de cartes à piocher SI le bonus
#                                 jumelles a été payé (Game.last_bonus_paid).
# DRAW_PER_COUNT[dweller_id]  -> ('dweller'|'type', id) : pioche 1 carte par
#                                 exemplaire déjà en forêt de ce dweller/type
#                                 (compté APRÈS la pose de la carte elle-même,
#                                 donc elle se compte aussi si applicable).
# REPLAY_ALWAYS = ensemble de dwellers qui font rejouer un tour complet,
#                 sans condition.
# REPLAY_IF_BONUS = ensemble de dwellers qui font rejouer un tour complet
#                    SI le bonus jumelles a été payé.
# TREE_DRAW_FIXED[tree_id]    -> pioche à la pose d'un Arbre (pas un dweller).
# ---------------------------------------------------------------------------

DRAW_FIXED = {
    D_BEECH_MARTEN: 1,   # Fouine
    D_POND_TURTLE: 1,    # Tortue cistude
    D_TREE_FERNS: 1,     # Fougère arborescente
    D_TAWNY_OWL: 1,      # Chouette hulotte : pioche de base, inconditionnelle
}

DRAW_IF_BONUS = {
    D_ROE_DEER: 1,       # Chevreuil
    D_FALLOW_DEER: 2,    # Daim
    D_HEDGEHOG: 1,       # Hérisson commun
    D_BROWN_BEAR: 1,     # Ours brun (nombre non confirmé par Mehdi, "?" dans
                          # cartes_effets.md ; 1 par défaut, cohérent avec le
                          # reste de cette table)
    D_TAWNY_OWL: 2,      # Chouette hulotte : 2 cartes DE PLUS si bonus jumelles payé
}

DRAW_PER_COUNT = {
    D_RED_FOX: ("dweller", D_HARE),   # Renard roux : 1 par Lièvre d'Europe
    D_WOLF: ("type", TY_DEER),        # Loup : 1 par Cervidé
}

REPLAY_ALWAYS = frozenset({D_EURASIAN_JAY})  # Geai des chênes, inconditionnel
REPLAY_IF_BONUS = frozenset({D_WOLF, D_BROWN_BEAR})  # bonus jumelles payé
# Sapin Douglas (dweller "SI bonus : rejoue un tour") pas encore ajouté ici :
# son dweller_id précis n'a pas été identifié dans le catalogue de Mehdi
# (nom générique "Sapin Douglas", à confirmer contre cards.py avant d'ajouter
# une entrée -- laissé en MANQUE pour ne pas deviner un id au hasard).

TREE_DRAW_FIXED = {
    T_BIRCH: 1,  # Bouleau : effet de pose, pioche 1 carte en plus du score
    T_BEECH: 1,  # Hêtre : effet de pose inconditionnel, pioche 1 carte (confirmé par Mehdi)
}

# Confirmé par Mehdi : un Arbre PEUT avoir un bonus jumelles (payé avec une
# carte de son propre symbole), contrairement à l'hypothèse précédente
# ("pas de bonus jumelles pour un arbre", cf. cartes_effets.md avant
# correction). Sapin Douglas et Chêne rejouent un tour SI payés avec le
# bonus ; aucun arbre ne rejoue un tour inconditionnellement.
TREE_REPLAY_ALWAYS = frozenset()
TREE_REPLAY_IF_BONUS = frozenset({T_DOUGLAS_FIR, T_OAK})

# -- Batch 2b : "jouer gratuitement une carte depuis la main" -------------
# Nécessite un sous-choix (Game.pending_effect, voir game.py). Confirmé par
# Mehdi pour ces 5 cartes ; Taupe et Raton laveur restent en MANQUE, leurs
# mécaniques ("jouer plusieurs cartes en payant leur coût", "défausser puis
# repiocher en vrac") ne rentrent pas dans ce pattern générique.
#
# FILTER_ANY_ANIMAL : n'importe quel dweller animal (pas Plante/Champignon).
# FILTER_TYPE[type_id] : dweller du type donné (ex. Bird pour la Lucane).
# FILTER_DWELLER[dweller_id] : exactement ce dweller (ex. Cerf élaphe).
FILTER_ANY_ANIMAL, FILTER_TYPE, FILTER_DWELLER = "any_animal", "type", "dweller"

# dweller_id ou tree_id déclencheur -> (filtre, arg, remaining) où remaining
# est 1 (une seule carte) ou None (autant que voulu, boucle jusqu'à "skip").
DWELLER_PLAY_FREE_IF_BONUS = {
    D_EUROPEAN_BADGER: (FILTER_ANY_ANIMAL, None, 1),        # Blaireau
    D_STAG_BEETLE: (FILTER_TYPE, TY_BIRD, 1),                # Lucane
    D_RED_DEER: (FILTER_DWELLER, DWELLER_ID["RED_DEER"], 1),  # Cerf élaphe
    DWELLER_ID["GNAT"]: (FILTER_TYPE, TY_BAT, None),          # Moustique
    D_SALAMANDER: (FILTER_ANY_ANIMAL, None, 1),              # Salamandre tachetée
}
TREE_PLAY_FREE_IF_BONUS = {
    T_SILVER_FIR: (FILTER_ANY_ANIMAL, None, 1),  # Sapin blanc
}

# Raton laveur (batch 2c) : place autant de cartes que voulu de la main
# directement sur la Grotte (1 pt chacune, `Forest.cave` déjà scoré), et
# pioche le même nombre. Résolu via un pending_effect de type dédié
# ("cave_choice") : le nombre de cartes est un vrai choix stratégique,
# donc exposé comme actions ("cave_discard", n) plutôt que décidé par une
# heuristique cachée.
D_RACCOON = DWELLER_ID["RACCOON"]
CAVE_CHOICE_DWELLERS = frozenset({D_RACCOON})

# Ours brun : à la pose, déplace INCONDITIONNELLEMENT toutes les cartes de
# la Clairière (Game.clearing) dans sa Grotte (confirmé par Mehdi). Pas de
# sous-choix (aucune carte à trier, elles y vont toutes), donc pas de
# pending_effect dédié : résolu directement dans _resolve_dweller_effect
# (game.py), qui a accès à self.clearing. Le tirage bonus jumelles (1 carte)
# et le rejeu de tour bonus jumelles rentrent dans les tables génériques
# DRAW_IF_BONUS / REPLAY_IF_BONUS ci-dessus.
CLEARING_TO_CAVE_DWELLERS = frozenset({D_BROWN_BEAR})

# Taupe (batch 2d) : "jouez immédiatement autant de cartes que souhaité en
# payant leur coût" — chaîne d'actions de pose normales (payantes), pas de
# sous-choix filtré ni de nombre fixé à l'avance. Confirmé par Mehdi :
# condition d'arrêt = on ne peut plus payer, ou choix stratégique de
# s'arrêter. Résolu via pending_effect de type "play_chain" : tant qu'il
# est actif, legal_actions() réexpose les poses payantes normales (sans
# piocher) plus skip_effect ; s'arrête seul quand plus aucune pose n'est
# affordable (seul skip_effect reste disponible).
D_MOLE = DWELLER_ID["MOLE"]
PLAY_CHAIN_DWELLERS = frozenset({D_MOLE})

# Champignons à effet permanent (brique 3) : tant qu'ils sont en jeu, ils
# se déclenchent à CHAQUE habitant posé par le même joueur qui remplit leur
# condition (pas seulement à leur propre pose). Un exemplaire supplémentaire
# du même champignon multiplie l'effet (comme Renard/Loup, DRAW_PER_COUNT).
# Portée : dwellers uniquement, pas les arbres — "carte portant le symbole
# arbre" (Girolle) est interprété comme "n'importe quel habitant" puisque
# seuls les habitants portent un symbole imprimé distinct de leur propre
# espèce dans ce moteur (voir card_symbol/game.py, qui renvoie None pour un
# Arbre). Compté APRÈS la pose de la carte elle-même : poser le champignon
# déclenche potentiellement son propre effet sur cette même pose, comme
# pour DRAW_PER_COUNT (même convention, à corriger si Mehdi infirme).
D_FLY_AGARIC = DWELLER_ID["FLY_AGARIC"]          # Amanite tue-mouches
D_PENNY_BUN = DWELLER_ID["PENNY_BUN"]            # Cèpe de Bordeaux
D_CHANTERELLE = DWELLER_ID["CHANTERELLE"]        # Girolle
D_PARASOL_MUSHROOM = DWELLER_ID["PARASOL_MUSHROOM"]  # Coulemelle

MUSHROOM_TRIGGER_ANIMAL = "animal"      # condition : la carte posée est animale
MUSHROOM_TRIGGER_POSITION = "position"  # condition : posée à une position donnée
MUSHROOM_TRIGGER_ANY_SYMBOL = "any_symbol"  # condition : n'importe quel habitant

MUSHROOM_TRIGGER = {
    D_FLY_AGARIC: (MUSHROOM_TRIGGER_ANIMAL, None),
    D_PENNY_BUN: (MUSHROOM_TRIGGER_POSITION, TOP),
    D_CHANTERELLE: (MUSHROOM_TRIGGER_ANY_SYMBOL, None),
    D_PARASOL_MUSHROOM: (MUSHROOM_TRIGGER_POSITION, BOTTOM),
}

BAT_IDS = frozenset(i for i, d in enumerate(DWELLERS) if "Bat" in d.types)
BUTTERFLY_IDS = tuple(i for i, d in enumerate(DWELLERS) if "Butterfly" in d.types)
BUTTERFLY_SLOT = {d: k for k, d in enumerate(BUTTERFLY_IDS)}
IS_BAT = tuple(i in BAT_IDS for i in range(N_DWELLERS))


# Points fixes des habitants qui ne dépendent de rien (table plate).
FLAT_POINTS = [0] * N_DWELLERS
FLAT_POINTS[D_EURASIAN_JAY] = 3
FLAT_POINTS[D_EUROPEAN_BADGER] = 2
FLAT_POINTS[D_POND_TURTLE] = 5
FLAT_POINTS[D_SQUEAKER] = 1
FLAT_POINTS[D_TAWNY_OWL] = 5

HORSE_CHESTNUT_POINTS = (0, 1, 4, 9, 16, 25, 36, 49)  # index = effectif, plafonné
SALAMANDER_POINTS = (0, 5, 15, 25)
FIREFLIES_POINTS = (0, 0, 10, 15, 20)
BUTTERFLY_POINTS = (0, 0, 3, 6, 12, 20, 35, 55, 80)


def _threshold(count, table):
    """Table indexée par effectif, plafonnée au dernier palier."""
    if count <= 0:
        return 0
    return table[count] if count < len(table) else table[-1]


class IllegalMove(Exception):
    pass


# ---------------------------------------------------------------------------
# Forêt
# ---------------------------------------------------------------------------


class Forest:
    """Forêt d'un joueur, avec compteurs maintenus à jour à chaque pose.

    Le scoring ne relit jamais `slots` : tout ce dont les règles ont besoin est
    déjà agrégé. `slots` ne sert qu'à la légalité des poses et au débogage.
    """

    __slots__ = (
        "species", "slots", "cave",
        "n_trees", "tree_count", "n_species",
        "dweller_count", "type_count",
        "n_dwellers", "bottom_total", "fully_occupied",
        "dwellers_on_tree", "occupied_positions",
        "silver_fir_dwellers", "chaffinch_on_beech", "squirrel_on_oak",
        "roe_deer_by_species", "bee_by_species", "n_bees",
        "symbol_count", "bat_species", "butterfly_count",
        "toad_bottom", "toad_scoring_cards",
        "bats_by_side", "dormice_by_side", "dormouse_hits",
    )

    def __init__(self):
        self.species = []                 # espèce de chaque arbre
        self.slots = []                   # [tree][pos] -> liste d'ids
        self.cave = 0

        self.n_trees = 0
        self.tree_count = [0] * N_TREES
        self.n_species = 0

        self.dweller_count = [0] * N_DWELLERS
        self.type_count = [0] * N_TYPES
        self.n_dwellers = 0
        self.bottom_total = 0
        self.fully_occupied = 0

        self.dwellers_on_tree = []        # nb d'habitants par arbre
        self.occupied_positions = []      # nb de positions occupées par arbre

        self.silver_fir_dwellers = 0
        self.chaffinch_on_beech = 0
        self.squirrel_on_oak = 0
        self.roe_deer_by_species = [0] * N_TREES
        self.bee_by_species = [0] * N_TREES
        self.n_bees = 0
        self.symbol_count = [0] * N_TREES  # cartes portant ce symbole d'arbre

        self.bat_species = [0] * N_DWELLERS
        self.butterfly_count = [0] * len(BUTTERFLY_IDS)

        self.toad_bottom = []             # crapauds au Bottom de chaque arbre
        self.toad_scoring_cards = 0       # crapauds appartenant à une paire

        self.bats_by_side = []            # [tree] -> [bats Left, bats Right]
        self.dormice_by_side = []
        self.dormouse_hits = 0            # loirs activés par une chauve-souris

    # -- construction ------------------------------------------------------

    def copy(self):
        f = Forest.__new__(Forest)
        f.species = self.species[:]
        f.slots = [s[:] if s.__class__ is list else s for s in self.slots]
        # slots est une liste de listes de listes : copie profonde nécessaire
        f.slots = [[lst[:] for lst in tree] for tree in self.slots]
        f.cave = self.cave
        f.n_trees = self.n_trees
        f.tree_count = self.tree_count[:]
        f.n_species = self.n_species
        f.dweller_count = self.dweller_count[:]
        f.type_count = self.type_count[:]
        f.n_dwellers = self.n_dwellers
        f.bottom_total = self.bottom_total
        f.fully_occupied = self.fully_occupied
        f.dwellers_on_tree = self.dwellers_on_tree[:]
        f.occupied_positions = self.occupied_positions[:]
        f.silver_fir_dwellers = self.silver_fir_dwellers
        f.chaffinch_on_beech = self.chaffinch_on_beech
        f.squirrel_on_oak = self.squirrel_on_oak
        f.roe_deer_by_species = self.roe_deer_by_species[:]
        f.bee_by_species = self.bee_by_species[:]
        f.n_bees = self.n_bees
        f.symbol_count = self.symbol_count[:]
        f.bat_species = self.bat_species[:]
        f.butterfly_count = self.butterfly_count[:]
        f.toad_bottom = self.toad_bottom[:]
        f.toad_scoring_cards = self.toad_scoring_cards
        f.bats_by_side = [b[:] for b in self.bats_by_side]
        f.dormice_by_side = [d[:] for d in self.dormice_by_side]
        f.dormouse_hits = self.dormouse_hits
        return f

    def add_tree(self, tree_id):
        if self.tree_count[tree_id] == 0:
            self.n_species += 1
        self.tree_count[tree_id] += 1
        self.n_trees += 1
        self.symbol_count[tree_id] += 1
        self.species.append(tree_id)
        self.slots.append([[], [], [], []])
        self.dwellers_on_tree.append(0)
        self.occupied_positions.append(0)
        self.toad_bottom.append(0)
        self.bats_by_side.append([0, 0])
        self.dormice_by_side.append([0, 0])
        return len(self.species) - 1

    def can_place(self, tree_idx, pos, dweller_id):
        """Règle officielle (rulebook FR confirmé par capture d'écran) :
        le côté (Top/Bottom/Left/Right imprimé sur la carte) doit
        correspondre à un slot vide, mais l'espèce de l'arbre ne restreint
        pas le placement. L'espèce imprimée sert au bonus de paiement par
        couleur et à la règle ROE_DEER - voir add_dweller.
        """
        if pos not in VALID_POS[dweller_id]:
            return False
        slot = self.slots[tree_idx][pos]
        if not slot:
            return True
        cap = SHARE_MAX[dweller_id]
        if cap == 0:
            return False
        if slot[0][0] != dweller_id:
            return False
        return cap < 0 or len(slot) < cap

    def legal_positions(self, dweller_id):
        """Itère les (tree_idx, pos) où cet habitant peut être posé.

        Toute position vide du bon côté convient, quelle que soit l'espèce
        de l'arbre (voir can_place) ; le côté lui-même reste fixé par la
        carte (VALID_POS).
        """
        valid_pos = VALID_POS[dweller_id]
        for i in range(len(self.species)):
            slots = self.slots[i]
            for pos in valid_pos:
                slot = slots[pos]
                if not slot:
                    yield i, pos
                    continue
                cap = SHARE_MAX[dweller_id]
                if cap == 0 or slot[0][0] != dweller_id:
                    continue
                if cap < 0 or len(slot) < cap:
                    yield i, pos

    def add_dweller(self, tree_idx, pos, dweller_id, symbol=None):
        """symbol : espèce imprimée sur la carte jouée (fixe, indépendante de
        l'arbre porteur - confirmé par le rulebook FR : le bonus de paiement
        et la règle ROE_DEER se basent sur cette couleur imprimée, pas sur
        l'arbre réel). Par défaut = espèce de l'arbre porteur, pour rester
        compatible avec les appels qui ne la précisent pas encore (tests,
        bench) ; dans ce cas le comportement est inchangé par rapport à
        avant. game.py la précise désormais avec la vraie valeur.

        Seuls symbol_count et roe_deer_by_species (donc uniquement la règle
        ROE_DEER) utilisent ce symbole imprimé. Toutes les autres règles
        positionnelles (CHAFFINCH "sur un Beech", RED_SQUIRREL "sur un Oak",
        SILVER_FIR "attaché à ce Silver Fir") continuent d'utiliser l'arbre
        réel, qu'on retrouve avec self.species[tree_idx] ci-dessous.
        """
        slot = self.slots[tree_idx][pos]
        if not slot:
            self.occupied_positions[tree_idx] += 1
            if self.occupied_positions[tree_idx] == 4:
                self.fully_occupied += 1
        slot.append((dweller_id, symbol))

        sp = self.species[tree_idx]          # arbre réel (attachement)
        sym = sp if symbol is None else symbol  # symbole imprimé (ROE_DEER)
        self.dwellers_on_tree[tree_idx] += 1
        self.n_dwellers += 1
        self.dweller_count[dweller_id] += 1
        self.symbol_count[sym] += 1
        for t in DWELLER_TYPE_IDS[dweller_id]:
            self.type_count[t] += 1

        if pos == BOTTOM:
            self.bottom_total += 1
        if sp == T_SILVER_FIR:
            self.silver_fir_dwellers += 1

        if IS_BAT[dweller_id]:
            self.bat_species[dweller_id] += 1
            if pos == LEFT or pos == RIGHT:
                side = 0 if pos == LEFT else 1
                self.bats_by_side[tree_idx][side] += 1
                if self.bats_by_side[tree_idx][side] == 1:
                    # active les loirs déjà posés en face
                    self.dormouse_hits += self.dormice_by_side[tree_idx][1 - side]
        elif dweller_id == D_FAT_DORMOUSE:
            side = 0 if pos == LEFT else 1
            self.dormice_by_side[tree_idx][side] += 1
            if self.bats_by_side[tree_idx][1 - side]:
                self.dormouse_hits += 1
        elif dweller_id == D_COMMON_TOAD:
            self.toad_bottom[tree_idx] += 1
            n = self.toad_bottom[tree_idx]
            if n == 2:
                self.toad_scoring_cards += 2
            elif n > 2:
                self.toad_scoring_cards += 1
        elif dweller_id == D_CHAFFINCH:
            if sp == T_BEECH:
                self.chaffinch_on_beech += 1
        elif dweller_id == D_RED_SQUIRREL:
            if sp == T_OAK:
                self.squirrel_on_oak += 1
        elif dweller_id == D_ROE_DEER:
            self.roe_deer_by_species[sym] += 1
        elif dweller_id == D_BEE:
            self.bee_by_species[sp] += 1
            self.n_bees += 1

        k = BUTTERFLY_SLOT.get(dweller_id)
        if k is not None:
            self.butterfly_count[k] += 1

    def undo_dweller(self, tree_idx, pos, dweller_id):
        """Annule le dernier add_dweller sur ce slot.

        Toutes les mises à jour incrémentales sont inversibles tant que
        l'annulation suit immédiatement la pose (ordre LIFO). C'est ce qui
        permet d'évaluer un coup candidat sans copier la forêt : poser,
        scorer, annuler coûte environ 5 us, contre 15 us pour une copie.
        """
        slot = self.slots[tree_idx][pos]
        assert slot and slot[-1][0] == dweller_id, "annulation hors ordre LIFO"
        _, symbol = slot.pop()
        if not slot:
            if self.occupied_positions[tree_idx] == 4:
                self.fully_occupied -= 1
            self.occupied_positions[tree_idx] -= 1

        sp = self.species[tree_idx]
        sym = sp if symbol is None else symbol
        self.dwellers_on_tree[tree_idx] -= 1
        self.n_dwellers -= 1
        self.dweller_count[dweller_id] -= 1
        self.symbol_count[sym] -= 1
        for t in DWELLER_TYPE_IDS[dweller_id]:
            self.type_count[t] -= 1

        if pos == BOTTOM:
            self.bottom_total -= 1
        if sp == T_SILVER_FIR:
            self.silver_fir_dwellers -= 1

        if IS_BAT[dweller_id]:
            self.bat_species[dweller_id] -= 1
            if pos == LEFT or pos == RIGHT:
                side = 0 if pos == LEFT else 1
                self.bats_by_side[tree_idx][side] -= 1
                if self.bats_by_side[tree_idx][side] == 0:
                    self.dormouse_hits -= self.dormice_by_side[tree_idx][1 - side]
        elif dweller_id == D_FAT_DORMOUSE:
            side = 0 if pos == LEFT else 1
            self.dormice_by_side[tree_idx][side] -= 1
            if self.bats_by_side[tree_idx][1 - side]:
                self.dormouse_hits -= 1
        elif dweller_id == D_COMMON_TOAD:
            n = self.toad_bottom[tree_idx]
            if n == 2:
                self.toad_scoring_cards -= 2
            elif n > 2:
                self.toad_scoring_cards -= 1
            self.toad_bottom[tree_idx] -= 1
        elif dweller_id == D_CHAFFINCH:
            if sp == T_BEECH:
                self.chaffinch_on_beech -= 1
        elif dweller_id == D_RED_SQUIRREL:
            if sp == T_OAK:
                self.squirrel_on_oak -= 1
        elif dweller_id == D_ROE_DEER:
            self.roe_deer_by_species[sym] -= 1
        elif dweller_id == D_BEE:
            self.bee_by_species[sp] -= 1
            self.n_bees -= 1

        k = BUTTERFLY_SLOT.get(dweller_id)
        if k is not None:
            self.butterfly_count[k] -= 1

    def undo_tree(self):
        """Annule le dernier add_tree (l'arbre doit être vide)."""
        tree_id = self.species.pop()
        assert not any(self.slots[-1]), "l'arbre porte encore des habitants"
        self.slots.pop()
        self.dwellers_on_tree.pop()
        self.occupied_positions.pop()
        self.toad_bottom.pop()
        self.bats_by_side.pop()
        self.dormice_by_side.pop()
        self.tree_count[tree_id] -= 1
        if self.tree_count[tree_id] == 0:
            self.n_species -= 1
        self.n_trees -= 1
        self.symbol_count[tree_id] -= 1

    def delta_dweller(self, tree_idx, pos, dweller_id,
                      linden_majority=True, tree_majority=True):
        """Gain de score immédiat d'une pose, sans muter durablement."""
        before = self.score(linden_majority, tree_majority)
        self.add_dweller(tree_idx, pos, dweller_id)
        after = self.score(linden_majority, tree_majority)
        self.undo_dweller(tree_idx, pos, dweller_id)
        return after - before

    def delta_tree(self, tree_id, linden_majority=True, tree_majority=True):
        before = self.score(linden_majority, tree_majority)
        self.add_tree(tree_id)
        after = self.score(linden_majority, tree_majority)
        self.undo_tree()
        return after - before

    # -- comptages exposés (utilisés par le scoring multijoueur) ------------

    def linden_count(self):
        return self.tree_count[T_LINDEN] + self.bee_by_species[T_LINDEN]

    def tree_count_with_modifiers(self):
        return self.n_trees + self.n_bees

    def butterfly_score(self):
        """Somme des sets de papillons, en forme close.

        Les sets construits en first-fit par le moteur de référence ont, une
        fois les effectifs triés, une taille égale au nombre d'espèces
        présentes en plus de r exemplaires. Aucune boucle dynamique nécessaire.
        """
        counts = self.butterfly_count
        top = max(counts)
        if top == 0:
            return 0
        total = 0
        for r in range(top):
            k = 0
            for c in counts:
                if c > r:
                    k += 1
            total += BUTTERFLY_POINTS[k] if k < len(BUTTERFLY_POINTS) else BUTTERFLY_POINTS[-1]
        return total

    # -- scoring -----------------------------------------------------------

    def score(self, linden_majority=True, tree_majority=True):
        tc = self.tree_count
        dc = self.dweller_count
        ty = self.type_count
        n_trees = self.n_trees
        species = self.n_species

        score = 0

        # --- Arbres ---
        score += tc[T_BIRCH]
        score += tc[T_LINDEN] * (3 if linden_majority else 1)
        beech = tc[T_BEECH] + self.bee_by_species[T_BEECH]
        if beech >= 4:
            score += 5 * tc[T_BEECH]
        score += 5 * tc[T_DOUGLAS_FIR]
        if species >= 8:
            score += 10 * tc[T_OAK]
        score += 2 * self.silver_fir_dwellers
        score += n_trees * tc[T_SYCAMORE]
        hc = tc[T_HORSE_CHESTNUT] + self.bee_by_species[T_HORSE_CHESTNUT]
        score += _threshold(hc, HORSE_CHESTNUT_POINTS)

        # --- Habitants à points fixes ---
        score += (3 * dc[D_EURASIAN_JAY] + 2 * dc[D_EUROPEAN_BADGER]
                  + 5 * dc[D_POND_TURTLE] + dc[D_SQUEAKER]
                  + 5 * dc[D_TAWNY_OWL])

        # --- Habitants dépendant d'un comptage de types ---
        score += 2 * dc[D_BLACKBERRIES] * ty[TY_PLANT]
        score += 2 * dc[D_BULLFINCH] * ty[TY_INSECT]
        score += 3 * dc[D_FALLOW_DEER] * ty[TY_CLOVEN]
        score += dc[D_GNAT] * ty[TY_BAT]
        score += 3 * dc[D_GOSHAWK] * ty[TY_BIRD]
        score += 2 * dc[D_HEDGEHOG] * ty[TY_BUTTERFLY]
        score += dc[D_RED_DEER] * (n_trees + ty[TY_PLANT])
        score += dc[D_STAG_BEETLE] * ty[TY_PAWED]
        score += 6 * dc[D_TREE_FERNS] * ty[TY_AMPHIBIAN]
        score += 5 * dc[D_WOLF] * ty[TY_DEER]

        # --- Habitants dépendant d'un comptage de noms ---
        score += dc[D_HARE] * dc[D_HARE]
        score += 2 * dc[D_RED_FOX] * dc[D_HARE]
        score += 5 * dc[D_TREE_FROG] * dc[D_GNAT]
        if dc[D_ROE_DEER]:
            score += 10 * dc[D_LYNX]
        if dc[D_SQUEAKER]:
            score += 10 * dc[D_WILD_BOAR]
        if species >= 8:
            score += 10 * dc[D_STRAWBERRIES]
        if self.tree_count_with_modifiers() >= 10:
            score += 10 * dc[D_MOSS]
        if tree_majority:
            score += 10 * dc[D_WOODPECKER]

        # --- Habitants dépendant de la position ---
        # fully_occupied = nombre d'ARBRES dont les 4 côtés (Top/Bottom/
        # Left/Right) sont occupés, pas un booléen "toute la forêt l'est"
        # (confirmé par Mehdi : la Fouine marque 5 pts par arbre entièrement
        # occupé, donc 5 × Fouines × arbres entièrement occupés).
        score += 5 * dc[D_BEECH_MARTEN] * self.fully_occupied
        score += 2 * dc[D_WOOD_ANT] * self.bottom_total
        score += 5 * self.chaffinch_on_beech
        score += 5 * self.squirrel_on_oak
        score += 15 * self.dormouse_hits
        score += 5 * self.toad_scoring_cards

        # ROE_DEER : 3 points par carte portant le même symbole d'arbre
        if dc[D_ROE_DEER]:
            rd = self.roe_deer_by_species
            sym = self.symbol_count
            acc = 0
            for s in range(N_TREES):
                if rd[s]:
                    acc += rd[s] * sym[s]
            score += 3 * acc

        # --- Sets ---
        score += _threshold(dc[D_SALAMANDER], SALAMANDER_POINTS)
        score += _threshold(dc[D_FIREFLIES], FIREFLIES_POINTS)
        if len([1 for b in BAT_IDS if dc[b]]) >= 3:
            score += 5 * sum(dc[b] for b in BAT_IDS)
        score += self.butterfly_score()

        # --- Grotte ---
        score += self.cave
        return score


def score_players(forests):
    """Scores d'une partie complète, avec les deux majorités correctement
    résolues entre joueurs. C'est la seule façon d'obtenir des scores
    comparables à une vraie partie : en solo, LINDEN et le pic épeiche sont
    gratuitement maximaux et gonflent le total.
    """
    if not forests:
        return []
    max_linden = max(f.linden_count() for f in forests)
    max_trees = max(f.tree_count_with_modifiers() for f in forests)
    return [
        f.score(
            linden_majority=f.linden_count() >= max_linden,
            tree_majority=f.tree_count_with_modifiers() >= max_trees,
        )
        for f in forests
    ]
