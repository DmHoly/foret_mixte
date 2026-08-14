"""
Données de cartes du jeu de base de Forêt Mixte (Forest Shuffle).

FICHIER GÉNÉRÉ. Ne pas éditer à la main.
Source : soldag/forest-shuffle-scoring, extraction via tools/gen_cards.py.
Vérifié contre src/game/__tests__/deck.test.ts : 66 arbres, 184 moitiés
d'habitants pour la boîte de base.
"""

from typing import NamedTuple


class TreeCard(NamedTuple):
    name: str
    cost: int
    copies: int


class DwellerVariant(NamedTuple):
    """Une moitié de carte physique : espèce + symbole d'arbre + position."""

    tree: str
    position: str
    copies: int


class DwellerCard(NamedTuple):
    name: str
    cost: int
    types: tuple
    variants: tuple


POSITIONS = ("Top", "Bottom", "Left", "Right")

TREES = (
    TreeCard("BEECH", cost=1, copies=10),
    TreeCard("BIRCH", cost=0, copies=10),
    TreeCard("DOUGLAS_FIR", cost=2, copies=7),
    TreeCard("HORSE_CHESTNUT", cost=2, copies=11),
    TreeCard("LINDEN", cost=1, copies=9),
    TreeCard("OAK", cost=2, copies=7),
    TreeCard("SILVER_FIR", cost=2, copies=6),
    TreeCard("SYCAMORE", cost=2, copies=6),
)

DWELLERS = (
    DwellerCard(
        "BARBASTELLE_BAT",
        cost=1,
        types=("Bat",),
        variants=(
            DwellerVariant("HORSE_CHESTNUT", "Left", 1),
            DwellerVariant("OAK", "Right", 1),
            DwellerVariant("SILVER_FIR", "Left", 1),
        ),
    ),
    DwellerCard(
        "BECHSTEINS_BAT",
        cost=1,
        types=("Bat",),
        variants=(
            DwellerVariant("BEECH", "Left", 1),
            DwellerVariant("BIRCH", "Right", 1),
            DwellerVariant("OAK", "Left", 1),
        ),
    ),
    DwellerCard(
        "BEECH_MARTEN",
        cost=1,
        types=("PawedAnimal",),
        variants=(
            DwellerVariant("BEECH", "Left", 1),
            DwellerVariant("HORSE_CHESTNUT", "Right", 2),
            DwellerVariant("OAK", "Right", 1),
            DwellerVariant("SYCAMORE", "Left", 1),
        ),
    ),
    DwellerCard(
        "BLACKBERRIES",
        cost=0,
        types=("Plant",),
        variants=(
            DwellerVariant("BEECH", "Bottom", 1),
            DwellerVariant("BIRCH", "Bottom", 1),
            DwellerVariant("SILVER_FIR", "Bottom", 1),
        ),
    ),
    DwellerCard(
        "BROWN_BEAR",
        cost=3,
        types=("PawedAnimal",),
        variants=(
            DwellerVariant("BEECH", "Right", 1),
            DwellerVariant("HORSE_CHESTNUT", "Right", 1),
            DwellerVariant("LINDEN", "Left", 1),
        ),
    ),
    DwellerCard(
        "BROWN_LONG_EARED_BAT",
        cost=1,
        types=("Bat",),
        variants=(
            DwellerVariant("BEECH", "Right", 1),
            DwellerVariant("SYCAMORE", "Left", 2),
        ),
    ),
    DwellerCard(
        "BULLFINCH",
        cost=1,
        types=("Bird",),
        variants=(
            DwellerVariant("DOUGLAS_FIR", "Top", 3),
            DwellerVariant("SILVER_FIR", "Top", 1),
        ),
    ),
    DwellerCard(
        "CAMBERWELL_BEAUTY",
        cost=0,
        types=("Butterfly", "Insect"),
        variants=(
            DwellerVariant("BIRCH", "Top", 1),
            DwellerVariant("HORSE_CHESTNUT", "Top", 1),
            DwellerVariant("SYCAMORE", "Top", 2),
        ),
    ),
    DwellerCard(
        "CHAFFINCH",
        cost=1,
        types=("Bird",),
        variants=(
            DwellerVariant("BEECH", "Top", 1),
            DwellerVariant("BIRCH", "Top", 1),
            DwellerVariant("SYCAMORE", "Top", 2),
        ),
    ),
    DwellerCard(
        "CHANTERELLE",
        cost=2,
        types=("Mushroom",),
        variants=(
            DwellerVariant("BIRCH", "Bottom", 1),
            DwellerVariant("SILVER_FIR", "Bottom", 1),
        ),
    ),
    DwellerCard(
        "COMMON_TOAD",
        cost=0,
        types=("Amphibian",),
        variants=(
            DwellerVariant("BEECH", "Bottom", 1),
            DwellerVariant("DOUGLAS_FIR", "Bottom", 1),
            DwellerVariant("HORSE_CHESTNUT", "Bottom", 1),
            DwellerVariant("OAK", "Bottom", 1),
            DwellerVariant("SILVER_FIR", "Bottom", 1),
            DwellerVariant("SYCAMORE", "Bottom", 1),
        ),
    ),
    DwellerCard(
        "EURASIAN_JAY",
        cost=1,
        types=("Bird",),
        variants=(
            DwellerVariant("BIRCH", "Top", 2),
            DwellerVariant("HORSE_CHESTNUT", "Top", 1),
            DwellerVariant("SYCAMORE", "Top", 1),
        ),
    ),
    DwellerCard(
        "EUROPEAN_BADGER",
        cost=1,
        types=("PawedAnimal",),
        variants=(
            DwellerVariant("DOUGLAS_FIR", "Right", 2),
            DwellerVariant("HORSE_CHESTNUT", "Left", 2),
        ),
    ),
    DwellerCard(
        "EUROPEAN_FAT_DORMOUSE",
        cost=1,
        types=("PawedAnimal",),
        variants=(
            DwellerVariant("BEECH", "Left", 1),
            DwellerVariant("DOUGLAS_FIR", "Right", 1),
            DwellerVariant("OAK", "Right", 1),
            DwellerVariant("SILVER_FIR", "Left", 1),
        ),
    ),
    DwellerCard(
        "EUROPEAN_HARE",
        cost=0,
        types=("PawedAnimal",),
        variants=(
            DwellerVariant("BEECH", "Left", 1),
            DwellerVariant("BIRCH", "Left", 2),
            DwellerVariant("BIRCH", "Right", 1),
            DwellerVariant("LINDEN", "Left", 1),
            DwellerVariant("LINDEN", "Right", 1),
            DwellerVariant("OAK", "Left", 1),
            DwellerVariant("SILVER_FIR", "Left", 1),
            DwellerVariant("SILVER_FIR", "Right", 1),
            DwellerVariant("SYCAMORE", "Right", 2),
        ),
    ),
    DwellerCard(
        "FALLOW_DEER",
        cost=2,
        types=("ClovenhoofedAnimal", "Deer"),
        variants=(
            DwellerVariant("BIRCH", "Right", 1),
            DwellerVariant("LINDEN", "Left", 2),
            DwellerVariant("SYCAMORE", "Right", 1),
        ),
    ),
    DwellerCard(
        "FIREFLIES",
        cost=0,
        types=("Insect",),
        variants=(
            DwellerVariant("BEECH", "Bottom", 1),
            DwellerVariant("DOUGLAS_FIR", "Bottom", 1),
            DwellerVariant("LINDEN", "Bottom", 1),
            DwellerVariant("SYCAMORE", "Bottom", 1),
        ),
    ),
    DwellerCard(
        "FIRE_SALAMANDER",
        cost=1,
        types=("Amphibian",),
        variants=(
            DwellerVariant("DOUGLAS_FIR", "Bottom", 1),
            DwellerVariant("HORSE_CHESTNUT", "Bottom", 1),
            DwellerVariant("LINDEN", "Bottom", 1),
        ),
    ),
    DwellerCard(
        "FLY_AGARIC",
        cost=2,
        types=("Mushroom",),
        variants=(
            DwellerVariant("OAK", "Bottom", 1),
            DwellerVariant("SILVER_FIR", "Bottom", 1),
        ),
    ),
    DwellerCard(
        "GNAT",
        cost=0,
        types=("Insect",),
        variants=(
            DwellerVariant("BIRCH", "Left", 1),
            DwellerVariant("HORSE_CHESTNUT", "Right", 1),
            DwellerVariant("OAK", "Right", 1),
        ),
    ),
    DwellerCard(
        "GOSHAWK",
        cost=2,
        types=("Bird",),
        variants=(
            DwellerVariant("DOUGLAS_FIR", "Top", 1),
            DwellerVariant("OAK", "Top", 1),
            DwellerVariant("SILVER_FIR", "Top", 2),
        ),
    ),
    DwellerCard(
        "GREATER_HORSESHOE_BAT",
        cost=1,
        types=("Bat",),
        variants=(
            DwellerVariant("BEECH", "Left", 1),
            DwellerVariant("LINDEN", "Right", 2),
        ),
    ),
    DwellerCard(
        "GREAT_SPOTTED_WOODPECKER",
        cost=1,
        types=("Bird",),
        variants=(
            DwellerVariant("DOUGLAS_FIR", "Top", 1),
            DwellerVariant("LINDEN", "Top", 3),
        ),
    ),
    DwellerCard(
        "HEDGEHOG",
        cost=1,
        types=("PawedAnimal",),
        variants=(
            DwellerVariant("BEECH", "Bottom", 1),
            DwellerVariant("HORSE_CHESTNUT", "Bottom", 1),
            DwellerVariant("OAK", "Bottom", 1),
        ),
    ),
    DwellerCard(
        "LARGE_TORTOISESHELL",
        cost=0,
        types=("Butterfly", "Insect"),
        variants=(
            DwellerVariant("BEECH", "Top", 1),
            DwellerVariant("SILVER_FIR", "Top", 2),
            DwellerVariant("SYCAMORE", "Top", 1),
        ),
    ),
    DwellerCard(
        "LYNX",
        cost=1,
        types=("PawedAnimal",),
        variants=(
            DwellerVariant("BEECH", "Right", 1),
            DwellerVariant("DOUGLAS_FIR", "Left", 2),
            DwellerVariant("HORSE_CHESTNUT", "Left", 1),
            DwellerVariant("LINDEN", "Right", 1),
            DwellerVariant("SILVER_FIR", "Right", 1),
        ),
    ),
    DwellerCard(
        "MOLE",
        cost=2,
        types=("PawedAnimal",),
        variants=(
            DwellerVariant("OAK", "Bottom", 1),
            DwellerVariant("SYCAMORE", "Bottom", 1),
        ),
    ),
    DwellerCard(
        "MOSS",
        cost=0,
        types=("Plant",),
        variants=(
            DwellerVariant("DOUGLAS_FIR", "Bottom", 1),
            DwellerVariant("LINDEN", "Bottom", 2),
        ),
    ),
    DwellerCard(
        "PARASOL_MUSHROOM",
        cost=2,
        types=("Mushroom",),
        variants=(
            DwellerVariant("HORSE_CHESTNUT", "Bottom", 1),
            DwellerVariant("SILVER_FIR", "Bottom", 1),
        ),
    ),
    DwellerCard(
        "PEACOCK_BUTTERFLY",
        cost=0,
        types=("Butterfly", "Insect"),
        variants=(
            DwellerVariant("HORSE_CHESTNUT", "Top", 1),
            DwellerVariant("LINDEN", "Top", 1),
            DwellerVariant("OAK", "Top", 1),
            DwellerVariant("SILVER_FIR", "Top", 1),
        ),
    ),
    DwellerCard(
        "PENNY_BUN",
        cost=2,
        types=("Mushroom",),
        variants=(
            DwellerVariant("DOUGLAS_FIR", "Bottom", 2),
        ),
    ),
    DwellerCard(
        "POND_TURTLE",
        cost=2,
        types=("Amphibian",),
        variants=(
            DwellerVariant("BIRCH", "Bottom", 1),
            DwellerVariant("SYCAMORE", "Bottom", 1),
        ),
    ),
    DwellerCard(
        "PURPLE_EMPEROR",
        cost=0,
        types=("Butterfly", "Insect"),
        variants=(
            DwellerVariant("BIRCH", "Top", 1),
            DwellerVariant("HORSE_CHESTNUT", "Top", 2),
            DwellerVariant("LINDEN", "Top", 1),
        ),
    ),
    DwellerCard(
        "RACCOON",
        cost=1,
        types=("PawedAnimal",),
        variants=(
            DwellerVariant("BIRCH", "Right", 1),
            DwellerVariant("DOUGLAS_FIR", "Left", 1),
            DwellerVariant("SILVER_FIR", "Left", 1),
            DwellerVariant("SILVER_FIR", "Right", 1),
        ),
    ),
    DwellerCard(
        "RED_DEER",
        cost=2,
        types=("ClovenhoofedAnimal", "Deer"),
        variants=(
            DwellerVariant("HORSE_CHESTNUT", "Right", 2),
            DwellerVariant("LINDEN", "Left", 1),
            DwellerVariant("OAK", "Right", 1),
            DwellerVariant("SILVER_FIR", "Left", 1),
        ),
    ),
    DwellerCard(
        "RED_FOX",
        cost=2,
        types=("PawedAnimal",),
        variants=(
            DwellerVariant("BEECH", "Left", 1),
            DwellerVariant("DOUGLAS_FIR", "Right", 1),
            DwellerVariant("LINDEN", "Left", 2),
            DwellerVariant("OAK", "Right", 1),
        ),
    ),
    DwellerCard(
        "RED_SQUIRREL",
        cost=0,
        types=("PawedAnimal",),
        variants=(
            DwellerVariant("BEECH", "Top", 1),
            DwellerVariant("DOUGLAS_FIR", "Top", 1),
            DwellerVariant("HORSE_CHESTNUT", "Top", 1),
            DwellerVariant("OAK", "Top", 1),
        ),
    ),
    DwellerCard(
        "ROE_DEER",
        cost=2,
        types=("ClovenhoofedAnimal", "Deer"),
        variants=(
            DwellerVariant("BEECH", "Right", 1),
            DwellerVariant("BIRCH", "Right", 1),
            DwellerVariant("HORSE_CHESTNUT", "Right", 1),
            DwellerVariant("LINDEN", "Left", 1),
            DwellerVariant("SILVER_FIR", "Left", 1),
        ),
    ),
    DwellerCard(
        "SILVER_WASHED_FRITILLARY",
        cost=0,
        types=("Butterfly", "Insect"),
        variants=(
            DwellerVariant("BEECH", "Top", 1),
            DwellerVariant("OAK", "Top", 3),
        ),
    ),
    DwellerCard(
        "SQUEAKER",
        cost=0,
        types=("ClovenhoofedAnimal",),
        variants=(
            DwellerVariant("HORSE_CHESTNUT", "Left", 1),
            DwellerVariant("OAK", "Left", 1),
            DwellerVariant("OAK", "Right", 1),
            DwellerVariant("SYCAMORE", "Right", 1),
        ),
    ),
    DwellerCard(
        "STAG_BEETLE",
        cost=2,
        types=("Insect",),
        variants=(
            DwellerVariant("BIRCH", "Bottom", 1),
            DwellerVariant("SYCAMORE", "Bottom", 1),
        ),
    ),
    DwellerCard(
        "TAWNY_OWL",
        cost=2,
        types=("Bird",),
        variants=(
            DwellerVariant("BEECH", "Top", 2),
            DwellerVariant("BIRCH", "Top", 1),
            DwellerVariant("SYCAMORE", "Top", 1),
        ),
    ),
    DwellerCard(
        "TREE_FERNS",
        cost=1,
        types=("Plant",),
        variants=(
            DwellerVariant("HORSE_CHESTNUT", "Bottom", 1),
            DwellerVariant("LINDEN", "Bottom", 1),
            DwellerVariant("SILVER_FIR", "Bottom", 1),
        ),
    ),
    DwellerCard(
        "TREE_FROG",
        cost=0,
        types=("Amphibian",),
        variants=(
            DwellerVariant("LINDEN", "Bottom", 1),
            DwellerVariant("OAK", "Bottom", 2),
        ),
    ),
    DwellerCard(
        "VIOLET_CARPENTER_BEE",
        cost=1,
        types=("Insect",),
        variants=(
            DwellerVariant("DOUGLAS_FIR", "Left", 1),
            DwellerVariant("DOUGLAS_FIR", "Right", 2),
            DwellerVariant("SILVER_FIR", "Left", 1),
        ),
    ),
    DwellerCard(
        "WILD_BOAR",
        cost=2,
        types=("ClovenhoofedAnimal",),
        variants=(
            DwellerVariant("BIRCH", "Left", 1),
            DwellerVariant("DOUGLAS_FIR", "Right", 1),
            DwellerVariant("OAK", "Right", 1),
            DwellerVariant("SYCAMORE", "Left", 2),
        ),
    ),
    DwellerCard(
        "WILD_STRAWBERRIES",
        cost=0,
        types=("Plant",),
        variants=(
            DwellerVariant("BIRCH", "Bottom", 1),
            DwellerVariant("SYCAMORE", "Bottom", 2),
        ),
    ),
    DwellerCard(
        "WOLF",
        cost=3,
        types=("PawedAnimal",),
        variants=(
            DwellerVariant("DOUGLAS_FIR", "Left", 1),
            DwellerVariant("SILVER_FIR", "Right", 2),
            DwellerVariant("SYCAMORE", "Left", 1),
        ),
    ),
    DwellerCard(
        "WOOD_ANT",
        cost=0,
        types=("Insect",),
        variants=(
            DwellerVariant("BEECH", "Bottom", 2),
            DwellerVariant("BIRCH", "Bottom", 1),
        ),
    ),
)

# Partage de slot (soldag/forest-shuffle-scoring, modifiers.enablesSlotSharing).
# COMMON_TOAD : jusqu'à 2 crapauds sur le même slot Bottom.
# EUROPEAN_HARE : nombre illimité de lièvres sur le même slot.
SLOT_SHARING = {
    "COMMON_TOAD": 2,
    "EUROPEAN_HARE": None,
}

# Composition physique de la boîte de base : chaque carte habitant porte deux
# moitiés (Top+Bottom ou Left+Right). 48 cartes Top/Bottom + 44 Left/Right.
PHYSICAL_TOP_BOTTOM_CARDS = 48
PHYSICAL_LEFT_RIGHT_CARDS = 44
WINTER_CARDS = 3

