"""Extraction de features pour la fonction de valeur apprise (sklearn).

Un vecteur = l'état de forêt + main d'UN joueur observé, à un instant T
d'une partie. Cible associée (dans le générateur de dataset) = gain de
score restant jusqu'à la fin de la partie (pas le score final brut, voir
gen_value_dataset.py).

Colonnes, dans l'ordre (voir FEATURE_NAMES pour l'introspection) :
  - tree_prop par espèce (N_TREES = 8) : proportion des arbres de cette
    espèce parmi tous les arbres posés (somme = 1, ou 0 si aucun arbre)
  - dweller_prop par espèce (N_DWELLERS = 49) : proportion parmi tous les
    habitants posés
  - type_prop par type biologique (N_TYPES = 10) : proportion parmi tous
    les habitants posés (un habitant peut compter dans plusieurs types)
  - cave (1)
  - mushroom actif par sorte (4 : Amanite, Cèpe, Girolle, Coulemelle)
  - hand_size (1)
  - n_symboles_bonus_en_main (1) : nb de symboles distincts en main qui
    correspondent à un dweller/arbre en jeu ayant un effet bonus jumelles
    (proxy du "potentiel de combo différé")
  - has_unlimited_stacker_in_hand (1) : 1 si une carte à SHARE_MAX=-1
    (ex. Lièvre d'Europe) est en main
  - clearing_size (1) : nombre de cartes dans la Clairière (`Game.clearing`,
    partagée entre joueurs). Absente jusqu'ici alors que la Clairière est
    devenue le principal moteur d'inflation du score (voir README,
    "Calibration des scores") -- un modèle qui l'ignore ne peut pas
    distinguer un état où la prochaine pioche est connue et bon marché
    d'un état où elle est aveugle.
  - clearing_min_cost (1) : coût le plus bas parmi les cartes de la
    Clairière (0 si vide), même heuristique que `choose_draw_source` --
    proxy direct de "combien vaut la pioche garantie de ce tour".

Total = 8 + 49 + 10 + 1 + 4 + 1 + 1 + 1 + 1 + 1 = 77 colonnes.

Volontairement PAS de features d'avancement de partie (score actuel,
tours écoulés, cartes restantes dans le deck), ET volontairement des
PROPORTIONS plutôt que des comptes bruts pour les arbres/habitants/types
(session du 14/08, sur demande de Mehdi). Deux itérations d'analyse
(permutation_importance) ont montré que le modèle exploitait des proxys
d'avancement de partie comme raccourci -- d'abord les features explicites
(score, tours, deck restant), puis, une fois celles-ci retirées, le simple
VOLUME de cartes posées (n_trees_total corrélait à -0.76 avec la cible,
parce que la forêt ne fait que grossir au fil de la partie). Passer en
proportions neutralise cette fuite : le modèle voit la FORME de la forêt
(quelle composition relative) plutôt que sa TAILLE (combien de tours ont
passé). `cave`, `mushroom_*`, `hand_size` restent en compte brut : ce
sont des petits entiers bornés, moins susceptibles de servir d'horloge
déguisée que le nombre total d'arbres/habitants.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine as E
import game as G

MUSHROOM_DIDS = tuple(E.MUSHROOM_TRIGGER.keys())

# Dwellers/arbres ayant un effet bonus jumelles (batch 2b) : utilisés pour
# le proxy "potentiel de combo en main".
BONUS_DWELLER_IDS = (set(E.DRAW_IF_BONUS) | set(E.DWELLER_PLAY_FREE_IF_BONUS)
                     | set(E.REPLAY_IF_BONUS))
BONUS_TREE_IDS = set(E.TREE_PLAY_FREE_IF_BONUS)

UNLIMITED_STACKER_IDS = tuple(d for d, cap in enumerate(E.SHARE_MAX) if cap == -1)

FEATURE_NAMES = (
    [f"tree_prop_{E.TREE_NAME[i]}" for i in range(E.N_TREES)]
    + [f"dweller_prop_{E.DWELLER_NAME[i]}" for i in range(E.N_DWELLERS)]
    + [f"type_prop_{i}" for i in range(E.N_TYPES)]
    + ["cave"]
    + [f"mushroom_{E.DWELLER_NAME[d]}" for d in MUSHROOM_DIDS]
    + ["hand_size", "n_bonus_symbols_in_hand", "has_unlimited_stacker"]
    + ["clearing_size", "clearing_min_cost"]
)
N_FEATURES = len(FEATURE_NAMES)


def _hand_symbols(hand):
    """Ensemble des symboles imprimés (tree_id) présents en main, côté
    habitants uniquement (les arbres n'ont pas de symbole distinct)."""
    symbols = set()
    for card in hand:
        if card[0] != G.DWELLER:
            continue
        symbols.add(card[1][1])
        symbols.add(card[2][1])
    return symbols


def extract_features(game, player_index):
    """Vecteur de features pour game.players[player_index], à cet instant.

    Ne prend délibérément PAS en compte l'avancement de la partie ni la
    taille absolue de la forêt : voir la note au-dessus de FEATURE_NAMES.
    """
    player = game.players[player_index]
    forest = player.forest
    hand = player.hand

    n_trees_total = float(forest.n_trees)
    n_dwellers_total = float(sum(forest.dweller_count))

    tree_denom = n_trees_total if n_trees_total > 0 else 1.0
    dweller_denom = n_dwellers_total if n_dwellers_total > 0 else 1.0

    feats = [forest.tree_count[i] / tree_denom for i in range(E.N_TREES)]
    feats.extend(forest.dweller_count[i] / dweller_denom for i in range(E.N_DWELLERS))
    feats.extend(forest.type_count[i] / dweller_denom for i in range(E.N_TYPES))
    feats.append(float(forest.cave))
    feats.extend(float(forest.dweller_count[d]) for d in MUSHROOM_DIDS)

    hand_symbols = _hand_symbols(hand)
    n_bonus_symbols = 0
    for sym in hand_symbols:
        # un symbole "utile" est celui d'un dweller/arbre à bonus déjà en
        # jeu (sinon le symbole ne débloque rien d'actionnable maintenant)
        if any(forest.dweller_count[d] > 0 for d in BONUS_DWELLER_IDS):
            n_bonus_symbols += 1
            break
    has_stacker = 1.0 if any(
        card[0] == G.DWELLER and (card[1][0] in UNLIMITED_STACKER_IDS or card[2][0] in UNLIMITED_STACKER_IDS)
        for card in hand
    ) else 0.0

    feats.append(float(len(hand)))
    feats.append(float(n_bonus_symbols))
    feats.append(has_stacker)

    clearing = game.clearing
    feats.append(float(len(clearing)))
    feats.append(float(min((G.card_min_cost(c) for c in clearing), default=0)))

    assert len(feats) == N_FEATURES, (len(feats), N_FEATURES)
    return feats
