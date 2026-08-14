"""
Scorer de RÉFÉRENCE : transcription littérale de soldag/forest-shuffle-scoring.

Ce module n'est pas fait pour être rapide. Il est fait pour être évidemment
correct : chaque règle est une fonction séparée, écrite dans le même ordre et
avec les mêmes primitives que le TypeScript d'origine (countCardNames,
countCardTypes, countTreeSymbols, scoreSet, scoreByCardMajority).

Il sert d'oracle aux tests de `engine.py`. Si les deux divergent sur une forêt,
c'est `engine.py` qui a tort.

Correspondance des primitives :
  countCards(forest, filter, {ignoreModifiers})  -> _count_cards
  countCardNames(forest, names)                  -> ignoreModifiers=False
  countCardTypes(forest, types)                  -> ignoreModifiers=True
  countTreeSymbols(forest, symbols)              -> ignoreModifiers=True
  scoreSet(...)                                  -> scoré UNE fois par set
  scoreByCardMajority(...)                       -> comparaison inter-joueurs
"""

from cards import DWELLERS, POSITIONS, TREES

TREE_NAMES = [t.name for t in TREES]
DWELLER_TYPES = {d.name: frozenset(d.types) for d in DWELLERS}

BAT_NAMES = frozenset(d.name for d in DWELLERS if "Bat" in d.types)
BUTTERFLY_NAMES = frozenset(d.name for d in DWELLERS if "Butterfly" in d.types)

HORSE_CHESTNUT_POINTS = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49}
FIRE_SALAMANDER_POINTS = {1: 5, 2: 15, 3: 25}
FIREFLIES_POINTS = {2: 10, 3: 15, 4: 20}
BUTTERFLY_POINTS = {2: 3, 3: 6, 4: 12, 5: 20, 6: 35, 7: 55, 8: 80}


class RefForest:
    """Forêt sous forme naïve : liste d'arbres, chacun avec 4 slots de listes."""

    def __init__(self):
        self.trees = []  # [{"name": str, "slots": {pos: [dweller_name, ...]}}]
        self.cave = 0

    def add_tree(self, name):
        self.trees.append({"name": name, "slots": {p: [] for p in POSITIONS}})
        return len(self.trees) - 1

    def add_dweller(self, tree_idx, position, name):
        self.trees[tree_idx]["slots"][position].append(name)

    def dwellers(self):
        """(nom, position, index de l'arbre porteur)."""
        for i, t in enumerate(self.trees):
            for p in POSITIONS:
                for name in t["slots"][p]:
                    yield name, p, i


# --------------------------------------------------------------------------
# Primitives de comptage
# --------------------------------------------------------------------------


def _bee_bonus_on(forest, tree_indices):
    """modifiers.woodyPlantCount de VIOLET_CARPENTER_BEE.

    L'abeille fait compter son arbre porteur une fois de plus, mais uniquement
    si cet arbre passe déjà le filtre. C'est pour ça que le bonus dépend de
    l'ensemble d'arbres retenu et pas seulement de la présence de l'abeille.
    """
    bonus = 0
    for i in tree_indices:
        for p in POSITIONS:
            bonus += forest.trees[i]["slots"][p].count("VIOLET_CARPENTER_BEE")
    return bonus


def count_card_names(forest, names, ignore_modifiers=False):
    names = set(names)
    matching_trees = [i for i, t in enumerate(forest.trees) if t["name"] in names]
    count = len(matching_trees)
    count += sum(1 for n, _, _ in forest.dwellers() if n in names)
    if not ignore_modifiers:
        count += _bee_bonus_on(forest, matching_trees)
    return count


def count_card_types(forest, types, ignore_modifiers=True):
    types = set(types)
    matching_trees = (
        list(range(len(forest.trees))) if "Tree" in types else []
    )
    count = len(matching_trees)
    count += sum(1 for n, _, _ in forest.dwellers() if DWELLER_TYPES[n] & types)
    if not ignore_modifiers:
        count += _bee_bonus_on(forest, matching_trees)
    return count


def count_tree_symbols(forest, symbols):
    """Compte toutes les cartes portant l'un de ces symboles d'arbre.

    Un arbre porte son propre symbole. Une moitié d'habitant porte le symbole
    de l'arbre sur lequel elle peut être posée, donc celui de son arbre
    porteur. C'est ce que fait countTreeSymbols côté TypeScript, et c'est ce
    qui rend ROE_DEER beaucoup plus fort qu'un simple comptage d'arbres.
    """
    symbols = set(symbols)
    count = sum(1 for t in forest.trees if t["name"] in symbols)
    count += sum(
        1
        for _, _, ti in forest.dwellers()
        if forest.trees[ti]["name"] in symbols
    )
    return count


def count_tree_species(forest):
    return len({t["name"] for t in forest.trees})


def score_by_count(count, points_by_count):
    for threshold in sorted(points_by_count, reverse=True):
        if count >= threshold:
            return points_by_count[threshold]
    return 0


# --------------------------------------------------------------------------
# Règles collectives
# --------------------------------------------------------------------------


def score_bats(forest):
    """5 points PAR carte chauve-souris dès 3 espèces distinctes."""
    species = {n for n, _, _ in forest.dwellers() if n in BAT_NAMES}
    return 5 if len(species) >= 3 else 0


def score_butterflies_total(forest):
    """Constitution de sets en first-fit, scoré une seule fois pour le lot.

    Le TypeScript place chaque papillon dans le premier set qui ne contient
    pas déjà son espèce. Le résultat ne dépend que du multiset des effectifs
    par espèce : avec les effectifs triés, le set de rang r a pour taille le
    nombre d'espèces présentes en plus de r exemplaires.
    """
    sets = []
    for name, _, _ in forest.dwellers():
        if name not in BUTTERFLY_NAMES:
            continue
        for s in sets:
            if name not in s:
                s.add(name)
                break
        else:
            sets.append({name})
    return sum(BUTTERFLY_POINTS.get(len(s), 0) for s in sets)


def _fully_occupied_trees(forest):
    return sum(
        1 for t in forest.trees if all(t["slots"][p] for p in POSITIONS)
    )


def _bottom_dwellers(forest):
    return sum(len(t["slots"]["Bottom"]) for t in forest.trees)


# --------------------------------------------------------------------------
# Scoring complet
# --------------------------------------------------------------------------


def score_forest(forest, opponents=()):
    """Score total d'une forêt.

    `opponents` : autres forêts de la partie, nécessaires pour les deux cartes
    à majorité (LINDEN, GREAT_SPOTTED_WOODPECKER). En solo, laisser vide : le
    joueur est alors trivialement majoritaire, ce qui gonfle le score et n'est
    PAS représentatif d'une partie réelle.
    """
    all_forests = [forest, *opponents]

    linden_count = count_card_names(forest, ["LINDEN"])
    linden_majority = linden_count >= max(
        count_card_names(f, ["LINDEN"]) for f in all_forests
    )
    tree_count_mod = count_card_types(forest, ["Tree"], ignore_modifiers=False)
    tree_majority = tree_count_mod >= max(
        count_card_types(f, ["Tree"], ignore_modifiers=False) for f in all_forests
    )

    species = count_tree_species(forest)
    beech_count = count_card_names(forest, ["BEECH"])

    score = 0

    # --- Arbres ---
    for t in forest.trees:
        name = t["name"]
        if name == "BIRCH":
            score += 1
        elif name == "LINDEN":
            score += 3 if linden_majority else 1
        elif name == "BEECH":
            score += 5 if beech_count >= 4 else 0
        elif name == "DOUGLAS_FIR":
            score += 5
        elif name == "OAK":
            score += 10 if species >= 8 else 0
        elif name == "SILVER_FIR":
            score += 2 * sum(len(t["slots"][p]) for p in POSITIONS)
        elif name == "SYCAMORE":
            score += count_card_types(forest, ["Tree"])

    # HORSE_CHESTNUT : set, scoré une seule fois
    hc = count_card_names(forest, ["HORSE_CHESTNUT"])
    if hc:
        score += score_by_count(hc, HORSE_CHESTNUT_POINTS)

    # --- Habitants ---
    fully_occupied = _fully_occupied_trees(forest)
    bottom_total = _bottom_dwellers(forest)
    bat_points = score_bats(forest)
    seen_set_cards = set()

    for name, pos, ti in forest.dwellers():
        host = forest.trees[ti]["name"]

        if name in BAT_NAMES:
            score += bat_points
        elif name in BUTTERFLY_NAMES:
            pass  # traité en bloc plus bas
        elif name == "BEECH_MARTEN":
            score += 5 * fully_occupied
        elif name == "BLACKBERRIES":
            score += 2 * count_card_types(forest, ["Plant"])
        elif name == "BULLFINCH":
            score += 2 * count_card_types(forest, ["Insect"])
        elif name == "CHAFFINCH":
            score += 5 if host == "BEECH" else 0
        elif name == "COMMON_TOAD":
            toads = forest.trees[ti]["slots"]["Bottom"].count("COMMON_TOAD")
            score += 5 if toads > 1 else 0
        elif name == "EURASIAN_JAY":
            score += 3
        elif name == "EUROPEAN_BADGER":
            score += 2
        elif name == "EUROPEAN_FAT_DORMOUSE":
            opposite = "Right" if pos == "Left" else "Left"
            has_bat = any(
                d in BAT_NAMES for d in forest.trees[ti]["slots"][opposite]
            )
            score += 15 if has_bat else 0
        elif name == "EUROPEAN_HARE":
            score += count_card_names(forest, ["EUROPEAN_HARE"])
        elif name == "FALLOW_DEER":
            score += 3 * count_card_types(forest, ["ClovenhoofedAnimal"])
        elif name == "FIRE_SALAMANDER":
            if "FIRE_SALAMANDER" not in seen_set_cards:
                seen_set_cards.add("FIRE_SALAMANDER")
                n = count_card_names(forest, ["FIRE_SALAMANDER"])
                score += score_by_count(n, FIRE_SALAMANDER_POINTS)
        elif name == "FIREFLIES":
            if "FIREFLIES" not in seen_set_cards:
                seen_set_cards.add("FIREFLIES")
                n = count_card_names(forest, ["FIREFLIES"])
                score += score_by_count(n, FIREFLIES_POINTS)
        elif name == "GNAT":
            score += count_card_types(forest, ["Bat"])
        elif name == "GOSHAWK":
            score += 3 * count_card_types(forest, ["Bird"])
        elif name == "GREAT_SPOTTED_WOODPECKER":
            score += 10 if tree_majority else 0
        elif name == "HEDGEHOG":
            score += 2 * count_card_types(forest, ["Butterfly"])
        elif name == "LYNX":
            score += 10 if count_card_names(forest, ["ROE_DEER"]) > 0 else 0
        elif name == "MOSS":
            score += 10 if tree_count_mod >= 10 else 0
        elif name == "POND_TURTLE":
            score += 5
        elif name == "RED_DEER":
            score += count_card_types(forest, ["Tree", "Plant"])
        elif name == "RED_FOX":
            score += 2 * count_card_names(forest, ["EUROPEAN_HARE"])
        elif name == "RED_SQUIRREL":
            score += 5 if host == "OAK" else 0
        elif name == "ROE_DEER":
            score += 3 * count_tree_symbols(forest, [host])
        elif name == "SQUEAKER":
            score += 1
        elif name == "STAG_BEETLE":
            score += count_card_types(forest, ["PawedAnimal"])
        elif name == "TAWNY_OWL":
            score += 5
        elif name == "TREE_FERNS":
            score += 6 * count_card_types(forest, ["Amphibian"])
        elif name == "TREE_FROG":
            score += 5 * count_card_names(forest, ["GNAT"])
        elif name == "WILD_BOAR":
            score += 10 if count_card_names(forest, ["SQUEAKER"]) > 0 else 0
        elif name == "WILD_STRAWBERRIES":
            score += 10 if species >= 8 else 0
        elif name == "WOLF":
            score += 5 * count_card_types(forest, ["Deer"])
        elif name == "WOOD_ANT":
            score += 2 * bottom_total
        # BROWN_BEAR, MOLE, RACCOON, VIOLET_CARPENTER_BEE et les 4 champignons
        # valent 0 point : ce sont des cartes à effet de jeu, pas de score.
        # Vérifié dans le moteur de référence (score: () => 0).

    score += score_butterflies_total(forest)
    score += forest.cave  # RegularCave : 1 point par carte cachée
    return score
