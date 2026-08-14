"""Politique gloutonne "consciente de la force des cartes" : au lieu du
delta de score immédiat seul (`greedy_action`), et au lieu d'un coût de
paiement forfaitaire (`card_strength.py`'s force nette, coût × moyenne
globale), cette politique :

  1. évalue une carte candidate par SON delta de score exact (comme
     greedy_action, sans bruit, calculé par le moteur) PLUS une correction
     de potentiel stratégique = `marginal_brut(carte)` (table mesurée par
     retrait contrefactuel en fin de partie, voir card_strength.py) -- le
     delta immédiat ne voit pas la synergie future (ex. ROE_DEER compte
     des symboles posés plus tard), marginal_brut la capture.
  2. choisit le paiement en défaussant les cartes de la main dont
     `marginal_brut` (pas le coût facial) est le plus FAIBLE -- payer avec
     ses cartes stratégiquement les moins utiles, pas les plus chères.
     Confirmé par Mehdi (session du 14/08) : le coût réel d'une carte
     dépend de CE avec quoi on la paie, pas d'une moyenne forfaitaire.
  3. le bonus jumelles reste prioritaire dans le choix de paiement quand
     il est disponible sans changer les cartes défaussées de nature
     (préférence déjà présente dans `choose_payment`), parce qu'il
     augmente la valeur de la carte posée (rejeu, pioche, pose gratuite)
     sans coût suppélmentaire -- une carte payée "en couleur" vaut plus
     cher que la même carte payée sans bonus.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine as E
from card_values import DWELLER_MARGINAL_BRUT, TREE_MARGINAL_BRUT
from game import DWELLER, TREE, card_symbol

# --- table indexée par id, avec repli sur la moyenne globale pour toute
# carte absente de l'échantillon (jamais jouée dans les parties observées).
_all_vals = list(DWELLER_MARGINAL_BRUT.values()) + list(TREE_MARGINAL_BRUT.values())
AVG_VALUE = sum(_all_vals) / len(_all_vals)

DWELLER_VALUE = [DWELLER_MARGINAL_BRUT.get(E.DWELLER_NAME[d], AVG_VALUE)
                 for d in range(E.N_DWELLERS)]
TREE_VALUE = [TREE_MARGINAL_BRUT.get(E.TREE_NAME[t], AVG_VALUE)
              for t in range(E.N_TREES)]


def _card_worth(card):
    """Valeur stratégique d'une carte en main pour le choix de paiement :
    le max des deux moitiés (on jouerait la meilleure si on la gardait),
    sauf pour un Arbre (une seule valeur)."""
    if card[0] == TREE:
        return TREE_VALUE[card[1]]
    return max(DWELLER_VALUE[card[1][0]], DWELLER_VALUE[card[2][0]])


def strength_payment(hand, cost, preferred_symbol=None):
    """Comme `game.choose_payment`, mais classe les cartes NON préférées
    par `_card_worth` croissant (on défausse les plus faibles stratégiquement)
    plutôt que par coût facial décroissant."""
    if cost <= 0:
        return []

    def matches_preferred(i):
        if preferred_symbol is None:
            return False
        c = hand[i]
        if c[0] != DWELLER:
            return False
        return (card_symbol(c, 0) == preferred_symbol
                or card_symbol(c, 1) == preferred_symbol)

    ranked = sorted(
        range(len(hand)),
        key=lambda i: (0 if matches_preferred(i) else 1, _card_worth(hand[i])),
    )
    return ranked[:cost]


def strength_action(game, rng, epsilon=0.0, candidates=None, tree_bonus=6, w=1.0):
    """Comme `search.greedy_action`, mais le score d'un candidat ajoute
    `w * (valeur_stratégique_jouée - valeur_stratégique_défaussée)` au
    delta de score exact. Retourne (action, payment) -- l'appelant DOIT
    passer `payment` explicitement à `game.apply()`, sinon le moteur
    retombe sur son heuristique de paiement par défaut (coût facial).
    """
    actions = game.legal_actions()
    if len(actions) == 1:
        return actions[0], None
    if epsilon and rng.random() < epsilon:
        a = rng.choice(actions)
        return a, None

    plays = [a for a in actions if a[0] not in ("draw", "skip_effect")]
    if not plays:
        return actions[0], None
    if candidates and len(plays) > candidates:
        plays = rng.sample(plays, candidates)

    player = game.players[game.current]
    forest = player.forest
    hand = player.hand
    base = forest.score()
    n_trees = forest.n_trees

    best, best_score, best_payment = None, -1e9, None
    for a in plays:
        if a[0] == "cave_discard":
            score = float(a[1])
            payment = None
        else:
            # Le paiement doit référencer la main APRÈS le retrait de la
            # carte jouée (c'est ce que Game.apply() fait en interne) :
            # on retire le même index ici pour que les indices de paiement
            # retournés restent valides une fois passés à apply().
            hand_index, _half_index, _symbol = game._find_card(hand, a)
            payment_hand = hand[:hand_index] + hand[hand_index + 1:]
            if a[0] == "tree":
                tid = a[1]
                cost = E.TREE_COST[tid]
                payment = strength_payment(payment_hand, cost, preferred_symbol=None)
                paid_worth = sum(_card_worth(payment_hand[i]) for i in payment)
                delta = forest.delta_tree(tid) + tree_bonus / (1 + n_trees)
                score = delta + w * (TREE_VALUE[tid] - AVG_VALUE - (paid_worth - cost * AVG_VALUE))
            else:
                _, did, tree_idx, pos = a
                cost = E.DWELLER_COST[did]
                payment = strength_payment(payment_hand, cost, preferred_symbol=None)
                paid_worth = sum(_card_worth(payment_hand[i]) for i in payment)
                forest.add_dweller(tree_idx, pos, did)
                delta = forest.score() - base
                forest.undo_dweller(tree_idx, pos, did)
                score = delta + w * (DWELLER_VALUE[did] - AVG_VALUE - (paid_worth - cost * AVG_VALUE))
        if score > best_score:
            best, best_score, best_payment = a, score, payment

    if best_score <= 0 and len(hand) <= 7:
        from search import _fallback_action
        return _fallback_action(game), None
    return best, best_payment
