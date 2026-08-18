# Modèles de valeur — quel fichier est quoi

Plusieurs `.joblib`/`.npz` cohabitent dans `reference/` parce que chaque
tentative d'amélioration du modèle de valeur a été gardée comme point de
comparaison plutôt que supprimée (voir la philosophie générale du dépôt :
un résultat nul ou négatif reste un résultat, pas une erreur à effacer).
Ce fichier sert d'index pour ne pas s'y perdre.

## Fichiers vivants (chargés par défaut)

| Fichier | Rôle |
|---|---|
| `pairwise_model.joblib` | Modèle de valeur contrastif **linéaire**, utilisé par `value_policy.make_pairwise_hybrid_leaf_eval()` (donc par tout MCTS instancié sans `model_path` explicite) -- seul modèle qui se décompose en fonction de valeur par état, donc le seul utilisable comme `leaf_eval` MCTS pour l'instant. Réentraîné le 16/08 (2e fois) avec `gen_pairwise_dataset.py` + `train_pairwise_model.py` sous le code actuel : heuristique `choose_draw_source` **forte** + urgence Clairière (`CLEARING_URGENCY_BONUS`) dans `greedy_action`. |
| `pairwise_gbm_model.joblib` | Modèle de comparaison pairwise **Gradient Boosting**, utilisé par `value_policy.make_pairwise_gbm_tiebreak()` -- branché dans `search.greedy_action(..., tiebreak=...)` pour départager les candidats presque à égalité de gain exact (opt-in, PAS actif par défaut). Entraîné le 18/08 par `train_pairwise_gbm.py` sur le même dataset que le modèle linéaire. **Bat B 74/100 au gating** (voir "Comparateur pairwise dans greedy_action" plus bas) -- le bot le plus fort du dépôt à ce jour, mais utilisable uniquement pour des décisions réelles (trop coûteux pour un rollout MCTS en l'état, voir cette même section). |
| `pairwise_dataset.npz` | Dataset de paires (diff features -> diff gain réel) ayant servi à entraîner `pairwise_model.joblib` ET `pairwise_gbm_model.joblib` ci-dessus. 150 parties, 21468 paires, 78 features. |
| `value_model.joblib`, `value_dataset.npz` | Modèle de valeur **absolu** (MLP), approche antérieure au modèle contrastif. Toujours chargé par `value_policy.make_leaf_eval()`/`make_hybrid_leaf_eval()`, utilisé par `bench.py` comme point de comparaison. Limite connue : MAE (~12 pts) plus grande que l'écart réel entre deux coups candidats (~9 pts) -- voir la docstring de `gen_pairwise_dataset.py`, c'est justement pour corriger ce défaut que l'approche contrastive a été introduite. |

## Sauvegardes historiques (non chargées par défaut, gardées pour comparaison/rollback)

| Fichier | Contexte |
|---|---|
| `pairwise_model_pre_clairiere_urgence.joblib`, `pairwise_dataset_pre_clairiere_urgence.npz` | Le modèle tel qu'il était **avant** l'ajout de l'urgence Clairière (`CLEARING_URGENCY_BONUS`) à `greedy_action` -- entraîné sur des trajectoires générées sans cette urgence. Réutiliser ce modèle avec le `greedy_action` actuel recrée le même problème de distribution périmée : classement agrégé de D tombé de 63.3% à 51.1% dans le tournoi (voir plus bas), corrigé par le réentraînement suivant. |
| `pairwise_model_pre_clairiere_forte.joblib`, `pairwise_dataset_pre_clairiere_forte.npz` | Le modèle contrastif tel qu'il était **avant** le premier réentraînement du 16/08 -- entraîné sous l'ancienne heuristique de Clairière (toujours la carte la moins chère). C'est ce modèle périmé qui faisait perdre MCTS+forte face à Greedy+forte (12/30) ; le réentraînement (voir `bench_heuristics.py`) fait remonter le taux de victoire à ~59%. Gardé pour pouvoir reproduire la comparaison avant/après. |
| `pairwise_model_stale_backup.joblib`, `pairwise_dataset_stale_backup.npz` | Snapshot du 15/08 19h48, avant le réentraînement "sous les règles actuelles" de cette même journée (commit `233b14e`) -- prédate même l'ajout des features Clairière (`clearing_size`/`clearing_min_cost`) au vecteur de features. Conservé comme point de repère le plus ancien. |
| `pairwise_model_true_original_aligned.joblib`, `pairwise_model_retrain1_no_clearing_aligned.joblib` | Paire de modèles produits lors de l'expérience "ajouter les features Clairière au modèle de valeur" (commit `013edbc`, 15/08 19h57) : le premier reprend l'entraînement d'origine réaligné sur le nouveau schéma de features (colonnes Clairière à zéro) pour servir de témoin, le second est réentraîné avec les features Clairière réellement peuplées. Résultat **nul** (l'ajout de ces features n'a pas amélioré le jeu de façon mesurable) -- gardés comme trace de la tentative, pas comme candidats à adopter. |
| `pairwise_model_bootstrap_gen1.joblib`, `pairwise_dataset_bootstrap_gen1.npz` | Génération 1 de la piste bootstrap MCTS façon AlphaZero (18/08) -- entraîné sur des trajectoires **MCTS**, pas greedy. Meilleur R²/MAE offline jamais mesuré (0.336/4.40) mais **rejeté au gating** contre B (7/30) -- voir la section "Génération 1 du bootstrap" plus bas pour le détail. |

## Comment regénérer le modèle vivant

```
python reference/gen_pairwise_dataset.py [n_parties]   # -> reference/pairwise_dataset.npz
python reference/train_pairwise_model.py                # -> reference/pairwise_model.joblib
```

Les deux scripts utilisent le code actuel de `game.py`/`search.py` tel
quel (aucune heuristique n'y est figée en dur) : relancer cette paire de
commandes régénère toujours un modèle cohérent avec le comportement de jeu
du moment. C'est exactement la source du bug corrigé le 16/08 -- le modèle
vivant datait d'avant un changement de comportement par défaut (l'ancienne
heuristique de Clairière), pas d'un bug dans ces scripts.

Pour comparer deux modèles en tête-à-tête, `reference/bench_heuristics.py`
accepte n'importe quel `model_path` via `value_policy.load_pairwise_model`
sans toucher au fichier par défaut.

## Résultat de référence (16/08, `pairwise_model.joblib` réentraîné)

Tournoi rond-robin complet entre les 4 variantes {Greedy, MCTS(150it)} x
{Clairière forte, Clairière faible}, sièges alternés, `python
reference/bench_heuristics.py` :

| Match | Score | Écart moyen (SE) | Médiane |
|---|---|---|---|
| B vs A | 69/100 | +47.6 (7.9) | +43.5 |
| D vs C | 21/30 | +36.1 (14.2) | +22.5 |
| B vs C | 20/30 | +47.8 (15.5) | +10.0 |
| D vs B | 15/30 (nul) | -29.7 (14.9) | +0.5 |
| D vs A | 21/30 | -2.5 (11.7) | +11.5 |
| C vs A | 15/30 (nul) | -36.9 (18.6) | -1.5 |

Classement agrégé (victoires / parties jouées) : **B 65.0%, D 63.3%, C
37.8%, A 33.1%**.

Deux points notables par rapport au tournoi avec l'ancien modèle (D à
40% contre B, C à 73% contre A) :

- **D vs B est passé de 12/30 à 15/30** (quasi-parité) : le réentraînement
  comble l'essentiel de l'écart qui faisait perdre le bot par défaut face
  à Greedy+forte.
- **C (MCTS + Clairière faible) s'est dégradé face à A** (73% -> 50%). Le
  modèle vivant est maintenant entraîné sur des parties jouées sous la
  heuristique **forte** ; l'utiliser avec un MCTS qui joue lui-même en
  heuristique **faible** le place hors de sa distribution d'entraînement
  -- l'inverse du problème corrigé pour D. Sans conséquence en pratique
  (l'heuristique faible n'est plus utilisée nulle part par défaut), mais
  c'est une bonne illustration du principe : un modèle de valeur n'est
  fiable que sur les états qui ressemblent à ceux vus à l'entraînement.

## Mise à jour (16/08, après l'urgence Clairière dans `greedy_action`)

L'ajout de `CLEARING_URGENCY_BONUS` (voir `search.py`) a de nouveau périmé
`pairwise_model.joblib` (même mécanisme, nouvelle cause) : classement de D
tombé à 51.1%. Réentraînement identique à la procédure ci-dessus. Nouveau
tournoi :

| Match | Score | Écart moyen (SE) | Médiane |
|---|---|---|---|
| B vs A | 67/100 | +52.1 (9.7) | +54.0 |
| D vs C | 21/30 | +44.7 (13.2) | +21.0 |
| B vs C | 25/30 | +109.9 (21.8) | +69.5 |
| D vs B | 13/30 | -40.4 (16.4) | -4.5 |
| D vs A | 15/30 (nul) | -2.4 (14.3) | +2.5 |
| C vs A | 14/30 (nul) | -24.8 (14.8) | -4.0 |

Classement agrégé : **B 67.5%, D 54.4%, A 39.4%, C 30.0%**.

Contrairement à la fois précédente, le réentraînement ne ramène **pas**
D vs B à la parité (13/30, plutôt pire qu'avant : 15/30). Sur trois
tournois consécutifs (avec trois modèles différents), **B (Greedy +
Clairière forte + urgence) bat systématiquement D (MCTS)** en tête-à-tête
direct, alors que D domine largement C et A. Hypothèse la plus probable :
ce n'est plus un problème de distribution périmée (déjà corrigé) mais un
plafond du modèle de valeur linéaire lui-même (R²≈0.10, MAE≈7.5 pts) --
une fois la base gloutonne déjà très forte grâce aux deux heuristiques,
la recherche arborescente a besoin d'un signal d'évaluation plus fin que
ce que le modèle actuel peut fournir pour départager les branches et
convertir sa profondeur en avantage réel. Améliorer encore le rollout
gagnant ne suffit plus ; la prochaine piste sérieuse est le modèle de
valeur lui-même (features, non-linéarité, volume de données), pas une
nouvelle heuristique de décision.

**En pratique, B (`greedy_action` avec les deux heuristiques actives) est
aujourd'hui le bot le plus fort du dépôt**, plus simple et bien plus
rapide que MCTS.

## Deux pistes "force brute" testées contre B, toutes deux négatives (16/08)

Question de Mehdi : un MCTS sans heuristique, plus coûteux mais "plus
pur", ne devrait-il pas battre B par force brute ? Deux variantes
testées en tête-à-tête, résultat négatif dans les deux cas.

**MCTS pur (`leaf_eval=None`, rollout réel profond à 90 coups au lieu du
modèle appris)** : 1re partie gagnée par MCTS pur (355 vs 322, partie
anormalement longue à 181 tours) -- mais sur n=30, **MCTS pur perd
largement : 8/30 (27%)**, écart moyen -74.3 pts (SE 21.8). Le rollout
reste guidé par `greedy_action`, mais avec 25% de coups purs aléatoires
(`ROLLOUT_EPSILON`) déroulés sur ~80-180 coups jusqu'à la fin de partie :
le bruit accumulé noie le signal du premier coup à évaluer, et 80
itérations ne suffisent pas à le moyenner. Un raccourci (modèle appris +
rollout court) donne une estimation moins chère ET plus fiable par
itération, à budget de calcul comparable.

**Modèle de valeur absolu (MLP) réentraîné sous les heuristiques actuelles**
(voir section précédente : R²=0.945 mais MAE=19pts, pire que le modèle
pairwise ~7.5pts) : encore plus net, **D avec ce modèle perd 2/30 (7%)**
contre B, écart moyen -152.9 pts (SE 18.8) -- pire que le MCTS pur
sans aucun modèle. Un modèle mal calibré pour la tâche (bon R² sur la
magnitude globale, mauvais sur les écarts fins entre coups voisins)
n'est pas juste inutile, il *biaise* activement la recherche vers de
mauvaises branches -- pire qu'aucun modèle du tout.

Conclusion des deux essais : le facteur limitant de MCTS sur ce jeu n'est
ni le manque de profondeur de rollout ni le manque de "réalisme" de la
simulation -- c'est la précision de l'évaluation de branches voisines à
un même nœud. Un modèle linéaire modeste (R²≈0.10) mais correctement
formulé pour cette tâche précise (diff de features -> diff de gain, voir
`gen_pairwise_dataset.py`) bat largement un modèle bien plus expressif
(MLP, R²≈0.95) mal formulé pour elle. Améliorer MCTS demanderait un
modèle **pairwise non-linéaire** (donc reformuler la tâche : un MLP sur
la différence de features ne se décompose pas en fonction de valeur par
état comme le fait un modèle linéaire, voir la docstring de
`gen_pairwise_dataset.py`).

## Troisième essai : modèle pairwise NON linéaire (réseau siamois, 16/08)

Suite logique du constat ci-dessus : `train_pairwise_mlp.py` reformule
la tâche correctement (même MLP appliqué séparément à chaque état
comparé, entraîné pour que la différence de sortie approche le vrai
delta -- voir la docstring du script pour le détail). Hors ligne, net
progrès : **R²=0.70, MAE=4.2pts**, contre R²=0.10/MAE=7.5pts pour le
modèle linéaire.

En tête-à-tête contre B pourtant : **D avec ce modèle perd 1/30 (3%)**,
écart moyen -148.9 pts (SE 21.1) -- aussi mauvais que le MLP absolu mal
formulé, malgré une précision hors ligne bien supérieure au modèle
linéaire qui, lui, tient la comparaison face à B (13-15/30).

## Diagnostic (16/08) : une fuite de données, pas un problème d'extrapolation

Question de Mehdi face à cet échec surprenant (un modèle bien meilleur
hors ligne perd quand même) : dataset trop petit ? mauvaises features ?
Diagnostic dédié (`diagnose_pairwise_bias.py`, scratchpad, adapté de
`reference/diagnose_value_bias.py`) : comparer les prédictions à une
référence propre (moyenne de 5 rollouts, peu bruitée) sur des états
**frais**, tirés indépendamment du dataset d'entraînement.

| Groupe | MAE linéaire | MAE MLP |
|---|---|---|
| on-policy (états "normaux") | 3.48 | **10.48** |
| off-policy (branches spéculatives type MCTS) | 2.41 | **13.11** |

Signal clé : le MLP est déjà mauvais **sur des états on-policy tout à
fait ordinaires** (10.48, très loin du MAE=4.2 annoncé à l'entraînement)
-- pas seulement hors distribution. Cause identifiée : `train_pairwise_mlp.py`
(comme `train_pairwise_model.py` avant lui) découpait train/test **au
niveau des paires**, pas des parties. Plusieurs paires proviennent de la
MÊME partie (échantillonnées tous les 4 tours le long d'une trajectoire)
-- des paires très corrélées se retrouvaient des deux côtés du split. Le
modèle linéaire, peu expressif, ne peut pas vraiment exploiter cette
fuite ; un MLP à ~7000 paramètres, si.

**Correction** : `gen_pairwise_dataset.py` trace maintenant `game_idx`
par paire ; les deux scripts d'entraînement splittent par
`GroupShuffleSplit` sur les parties, jamais sur les paires (le MLP passe
même à un split à 3 niveaux train/val/test : les hyperparamètres sont
choisis sur val, le test reste intact jusqu'au chiffre final). Dataset
regénéré à 250 parties. Résultat honnête :

| Modèle | Test R² | Test MAE |
|---|---|---|
| Linéaire (Ridge) | 0.075 | 7.23 pts |
| MLP (meilleure config : h=(16,8), l2=0.3, la plus petite/régularisée) | **-0.030** | **7.98 pts** |

**Le MLP n'apporte rien du tout une fois évalué honnêtement** -- il est
même légèrement pire que le modèle linéaire. Le R²=0.70 précédent était
presque entièrement un artefact de fuite de données, pas un vrai signal
appris que le linéaire aurait raté. Pas la peine de retester en tête-à-tête
contre B : le diagnostic hors ligne suffit, aucune config testée
n'approche la précision du modèle linéaire actuel.

Réponse à la question initiale ("dataset trop petit, ou mauvaises
features ?") : ni l'un ni l'autre au sens propre -- les features
suffisent (le linéaire les exploite déjà correctement), et plus de
données n'aurait rien changé au symptôme (la fuite existait quelle que
soit la taille du dataset, elle vient de la méthode de split, pas du
volume). Le vrai plafond, confirmé maintenant plutôt que supposé : ces
78 features, combinées linéairement, captent déjà l'essentiel du signal
exploitable à cet horizon de prédiction -- une architecture plus riche
n'aide pas sans features plus riches en amont.

**Bilan des 3 tentatives d'amélioration du modèle de valeur cette
session : toutes négatives.** MCTS pur (8/30), MLP absolu (2/30), MLP
pairwise non linéaire (jamais même arrivé au stade du test en partie,
recalé hors ligne) -- **B reste, de loin, le bot le plus fort et le plus
simple du dépôt.**

## Quatrième essai : réduire le bruit de la cible (k_rollout), 16/08

Nouvelle question de Mehdi face à l'échec ci-dessus : dataset trop
petit, ou mauvaises features ? Ni l'un ni l'autre -- diagnostic
(`noise_vs_features_experiment.py`, scratchpad) : la cible de chaque
paire venait d'un SEUL rollout de 20 coups (25% de hasard dedans,
`ROLLOUT_EPSILON`), pas d'une moyenne. Moyenner 5 rollouts par candidat
(mêmes graines communes préservées à chaque répétition, donc l'annulation
de bruit entre candidats d'un même nœud tient toujours) : **R²
0.089->0.249, MAE 7.72->4.54** à features identiques, sur un échantillon
contrôlé. k=10 n'apporte pas de gain net mesuré en plus. Testé aussi :
features de progression (tours écoulés/deck restant/Hivers vus, sûres
pour le pairwise car identiques des deux côtés d'une paire au même tour)
-- **mathématiquement inertes pour le modèle linéaire** : `Xd = featA -
featB` annule exactement toute feature partagée par les deux candidats,
donc `RidgeCV` leur assigne un poids nul par construction (vérifié :
résultat strictement identique avec/sans, à graine de dataset égale).

`gen_pairwise_dataset.py` passe `k_rollout=5` par défaut ; dataset
officiel régénéré (250 parties). Réentraînement :

| Modèle | Test R² | Test MAE |
|---|---|---|
| Linéaire (k=5) | **0.254** | **4.50** (meilleur jamais mesuré) |
| MLP (k=5, split honnête) | 0.192 | 4.86 (toujours pire que le linéaire) |

**Mais en tête-à-tête contre B, D avec ce modèle linéaire amélioré perd
encore PLUS largement qu'avant : 9/30 (30%), écart -77.1 (SE 22.7)** --
contre 13-15/30 pour l'ancien modèle plus bruité (k=1, moins régularisé
que jamais : alpha=4.28 contre alpha=7.85 avant).

## Motif qui se confirme sur 6 tentatives : mieux hors ligne = pire en jeu

| Tentative | Offline (R²/MAE) | Vs B en vrai |
|---|---|---|
| Linéaire original (k=1) | 0.075 / 7.23 | 13-15/30 |
| MCTS pur (sans modèle) | -- | 8/30 |
| MLP absolu | 0.945 / 19.0 (biaisé, cible mal formulée) | 2/30 |
| MLP pairwise (fuite train/test) | 0.70 / 4.2 (faux) | 1/30 |
| MLP pairwise (split honnête) | -0.03 / 7.98 | (pas testé, déjà pire hors ligne) |
| Linéaire, k=5 (meilleur hors ligne mesuré) | 0.254 / 4.50 | **9/30 (pire qu'avant)** |

Aucune amélioration mesurée hors ligne ne s'est jamais traduite par une
meilleure performance réelle contre B -- au contraire, plus le modèle
est précis SUR SA DISTRIBUTION D'ENTRAÎNEMENT (auto-jeu greedy), plus il
semble se dégrader en jeu réel. Hypothèse retenue : un modèle mieux
ajusté à cette distribution devient plus confiant, donc plus dangereux,
sur les branches hors-distribution que MCTS explore délibérément (terme
d'exploration UCT) -- un modèle imprécis mais "prudent" (fortement
régularisé) semble égarer moins la recherche qu'un modèle précis mais
surconfiant hors de ce qu'il a vu à l'entraînement.

**Conclusion de cette branche d'investigation (6 tentatives, toutes
négatives, motif cohérent) : améliorer le modèle de valeur en optimisant
sa précision offline n'est pas la bonne piste pour faire progresser MCTS
sur ce jeu.** Une vraie percée demanderait une méthode différente --
entraîner sur des états réellement visités par la recherche MCTS
elle-même (bootstrap itératif façon AlphaZero), pas sur de l'auto-jeu
greedy -- un chantier de recherche à part entière, pas une correction
incrémentale. **B (Greedy + ciblage carte forte + urgence Clairière)
reste, avec une conviction plus forte que jamais, le meilleur bot du
dépôt.**

## Piste bootstrap MCTS façon AlphaZero

Documentée ici sur demande de Mehdi (16/08) -- explique la logique, les
étapes et le pourquoi. **Génération 1 implémentée et testée le 18/08, voir
le résultat tout en bas de cette section** -- négatif, cohérent avec les 6
tentatives précédentes.

### Le problème exact que ça cible

Toutes les tentatives précédentes entraînent le modèle de valeur sur des
états issus d'un **auto-jeu greedy** (`S.greedy_action`, avec un peu de
bruit `trajectory_epsilon`). Mais au moment où MCTS s'en sert vraiment,
il ne se contente pas de suivre du greedy : son terme d'exploration UCT
le pousse délibérément à visiter des branches qu'un joueur greedy
n'atteindrait jamais (coups plus faibles en apparence, mais pas encore
assez explorés pour être écartés avec confiance). Le modèle n'a jamais vu
ces états à l'entraînement -- et le motif observé sur 6 tentatives
suggère qu'un modèle mieux ajusté à SA distribution d'entraînement (donc
plus confiant) devient plus dangereux, pas moins, quand on l'interroge
en dehors de cette distribution. Améliorer le modèle SUR DES DONNÉES
GREEDY ne peut donc pas résoudre un problème qui vient de la distribution
elle-même, aussi précis que devienne le modèle sur cette distribution.

### L'idée : faire coïncider distribution d'entraînement et distribution d'usage

Au lieu d'entraîner sur "ce que ferait un joueur greedy", entraîner sur
"ce que MCTS lui-même visite et conclut" -- en utilisant MCTS (avec le
modèle courant) comme générateur de trajectoires ET comme source de
cible, puis en réentraînant le modèle dessus, en boucle. C'est le
principe d'AlphaZero/AlphaGo Zero : le réseau de valeur n'apprend jamais
directement les règles du jeu, il apprend à imiter/anticiper ce que la
recherche arborescente elle-même conclut après avoir cherché -- puis,
une fois le réseau meilleur, la recherche devient meilleure aussi (elle
part d'un a priori plus juste), ce qui produit des données d'entraînement
encore meilleures au tour suivant. C'est un cercle vertueux, pas une
correction ponctuelle -- d'où "bootstrap" (on s'auto-améliore par ses
propres résultats, sans nouvelle vérité extérieure).

### Étapes, adaptées à ce dépôt

1. **Auto-jeu avec le MCTS courant** (au lieu de `S.greedy_action` dans
   `gen_pairwise_dataset.py`) : faire jouer `search.MCTS` contre lui-même
   (ou contre B, pour garder de la diversité), avec le modèle de valeur
   actuel branché. Coût : MCTS est beaucoup plus lent que greedy (~7-8s
   par partie contre <0.1s), donc moins de parties par génération que ce
   qu'on fait aujourd'hui (150-250) -- probablement 30-50 pour commencer,
   avec un nombre d'itérations réduit (50-80 au lieu de 150) pour tenir
   dans un temps raisonnable.

2. **Cible d'entraînement, deux options possibles** :
   - Garder la même mécanique qu'aujourd'hui (paires de candidats à un
     même nœud, rollout court à seed commune) mais échantillonner les
     nœuds de départ le long d'une trajectoire **MCTS**, pas greedy --
     correction minimale, ne change que la distribution des états visités.
   - Plus proche d'AlphaZero : utiliser directement les statistiques de
     l'arbre de recherche lui-même comme cible (`child.value / child.visits`
     pour chaque action candidate à la racine, déjà calculé par
     `MCTS.choose()`, gratuit à extraire) au lieu de relancer un rollout
     séparé. C'est un vrai changement de méthode : le modèle apprend à
     reproduire le jugement de la recherche complète (des centaines de
     simulations) plutôt qu'un seul rollout tronqué -- cible bien moins
     bruitée par construction, sans avoir besoin du `k_rollout` qu'on
     vient d'introduire.

3. **Réentraîner** le modèle sur ces nouvelles données (infrastructure
   déjà là : `train_pairwise_model.py`/`train_pairwise_mlp.py`, split par
   partie déjà corrigé).

4. **Valider avant de promouvoir** : ne remplacer le modèle "officiel"
   que si la nouvelle génération bat mesurablement l'ancienne ET B en
   tête-à-tête (`bench_heuristics.py`) -- jamais sur la seule base d'une
   métrique offline, exactement la leçon de cette session. C'est le
   mécanisme de "gating" qu'AlphaGo Zero utilise aussi (un candidat ne
   remplace le meilleur réseau que s'il le bat sur un vrai match).

5. **Répéter** sur plusieurs générations jusqu'à stagnation ou victoire
   nette contre B.

### Pourquoi c'est un vrai chantier, pas une correction

- **Coût de calcul multiplié** : chaque génération demande de générer des
  parties avec MCTS (lent) plutôt que greedy (quasi gratuit), puis
  réentraîner, puis valider en tête-à-tête -- et recommencer plusieurs
  fois. Probablement des heures de calcul cumulées, pas des minutes.
- **Risque d'instabilité** : le bootstrap peut osciller ou régresser
  d'une génération à l'autre (over/under-fitting sur les données de LA
  génération précédente) -- d'où l'étape de gating obligatoire, pas
  optionnelle.
- **Information imparfaite** : ce jeu cache la main adverse et l'ordre du
  deck (`determinize()` dans `search.py`) -- la collecte de données doit
  rester cohérente avec la vue déterminisée de l'observateur, pas fuiter
  d'information cachée dans les features (sinon on retombe dans le piège
  que `determinize()` a justement été conçu pour éviter).
- **Aucune garantie de succès** : vu que 6 tentatives différentes ont
  toutes échoué cette session avec un motif cohérent, il est possible que
  même un bootstrap bien exécuté ne suffise pas sans aussi revoir
  l'architecture (réseau de valeur ET de politique combinés, comme
  AlphaZero, pas juste un scalaire de valeur) -- un chantier de plusieurs
  itérations, pas un essai unique.

Si cette piste est tentée un jour, commencer petit (une seule génération
de bootstrap, 30 parties, 50 itérations MCTS) pour valider que le
principe fonctionne avant d'investir dans plusieurs générations.

## Génération 1 du bootstrap : implémentée et testée, résultat négatif (18/08)

Exactement l'échelle recommandée ci-dessus : 30 parties, MCTS(50it.) des
deux côtés (`reference/gen_pairwise_dataset_bootstrap.py`, sortie
`pairwise_dataset_bootstrap_gen1.npz`). Correction minimale (option 1 de
la section précédente) : le mécanisme d'étiquetage des paires (candidats
à un même nœud, k=5 rollouts à seed commune, repris tel quel de
`gen_pairwise_dataset.py`) est inchangé -- seule la trajectoire qui fait
avancer la partie entre deux points d'échantillonnage change : MCTS avec
le modèle vivant en `leaf_eval` (`VP.make_pairwise_hybrid_leaf_eval`), au
lieu de l'auto-jeu greedy bruité utilisé par toutes les tentatives
précédentes. Les deux heuristiques de `greedy_action` (`choose_draw_source`
forte, `CLEARING_URGENCY_BONUS`) restent actives par défaut partout dans
ce pipeline (rollouts de labellisation, mini-rollout du leaf_eval hybride
des bots de self-play) -- **pas touchées**, contrairement à une session
antérieure qui les avait débranchées pour générer des données et avait vu
le classement du bot entraîné dessus s'effondrer (voir les fichiers
`*_pre_clairiere_urgence*`/`*_pre_clairiere_forte*` plus haut). 4640
paires collectées, ~5-6s/partie.

Réentraîné (`train_pairwise_model.py pairwise_dataset_bootstrap_gen1.npz
pairwise_model_bootstrap_gen1.joblib`) :

| Modèle | Test R² | Test MAE |
|---|---|---|
| Bootstrap gen1 (trajectoires MCTS) | **0.336** | **4.40** (meilleur jamais mesuré, tous essais confondus) |
| Linéaire k=5 (trajectoires greedy, référence précédente) | 0.254 | 4.50 |

Meilleur score hors ligne jamais obtenu sur ce projet -- cohérent avec
l'hypothèse de départ (échantillonner la distribution réellement visitée
par MCTS devrait donner un signal plus pertinent). Gaté contre B avant
toute promotion (`reference/gate_bootstrap_gen1.py`, MCTS(150it) avec ce
modèle vs `greedy_action` B, sièges alternés, n=30) :

**D(candidat bootstrap gen1) vs B : 7/30 (23%), écart moyen -72.9 (SE 21.2),
médiane -25.5.**

**Rejeté -- pas promu comme modèle vivant.** Pire que le modèle linéaire
k=5 déjà rejeté (9/30) et que le tout premier modèle linéaire k=1 (13-15/30),
malgré le meilleur R²/MAE jamais mesuré. Confirme le motif de la section
précédente sur une 7e tentative, avec un angle différent (distribution
d'entraînement corrigée, pas juste bruit de cible réduit) : le bootstrap
change QUELS ÉTATS sont vus à l'entraînement, mais le facteur limitant
identifié plus haut n'est pas la distribution des états -- c'est la
capacité du modèle **linéaire** lui-même à discriminer des coups voisins
avec la précision fine qu'exige UCT pour départager des branches proches,
quelle que soit la distribution sur laquelle il est ajusté. Une seule
génération ne suffit sans doute pas à trancher si le principe du bootstrap
lui-même est viable (une génération peut juste être bruitée), mais vu le
coût d'une génération (~4 min de génération + gating) pour un résultat
dans la même fourchette que tout ce qui a déjà échoué, ce n'est pas jugé
prioritaire d'enchaîner une génération 2 sans un changement d'architecture
(modèle non linéaire correctement formulé, jamais validé faute d'avoir
dépassé le stade offline -- voir la tentative MLP pairwise ci-dessus).

**Modèle vivant inchangé : B (Greedy + Clairière forte + urgence) reste
le bot le plus fort du dépôt.** `pairwise_dataset_bootstrap_gen1.npz` et
`pairwise_model_bootstrap_gen1.joblib` gardés comme trace de cette
tentative (philosophie du dépôt, voir en tête de ce fichier) ;
`gen_pairwise_dataset_bootstrap.py` et `gate_bootstrap_gen1.py` restent
réutilisables si une génération 2 est tentée un jour.

## Diagnostic : R²/MAE global mesure la mauvaise chose (18/08)

Question de Mehdi face au paradoxe ci-dessus (R²/MAE s'améliore à chaque
itération -- k=1 : 0.075/7.23, k=5 : 0.254/4.50, bootstrap gen1 :
0.336/4.40 -- alors que le résultat réel contre B empire dans le même
temps -- 13-15/30, 9/30, 7/30) : les métriques du modèle sont-elles
bonnes ? Diagnostic dédié (`reference/diagnose_pairwise_metrics.py`) :
stratifier la précision de SIGNE par tranche de `|diff réel|`, sur le
même split test que l'entraînement (jamais vu).

| Modèle | \|diff\|<3 (serré) | \|diff\|∈[3,8) | \|diff\|∈[8,20) | \|diff\|≥20 (large) | Gating vs B |
|---|---|---|---|---|---|
| k=1 | 59.2% (n=639) | 56.9% | 60.0% | 64.0% | 13-15/30 |
| k=5 (vivant) | **47.9%** (n=1519) | 63.7% | 74.9% | 79.3% | 9/30 |
| Bootstrap gen1 | 58.4% (n=245) | 61.6% | 82.2% | **92.6%** | 7/30 |

**CORRECTIF (même jour, après coup) :** le tableau ci-dessus a d'abord
été publié avec des chiffres erronés sur la tranche serrée (k=1 77.8%,
k=5 69.4%, bootstrap 68.5%) -- bug dans la métrique, pas dans les
modèles. ~23% des paires ont un `|diff| réel` EXACTEMENT nul (candidats
dupliqués, ex. deux exemplaires identiques d'une carte en main -> même
état, même résultat de rollout). Un modèle linéaire SANS INTERCEPT les
"réussit" gratuitement par construction (`w.(fi-fi) = 0`, donc
`sign(0)==sign(0)`) quels que soient ses poids appris -- ça n'a rien à
voir avec sa capacité à discriminer quoi que ce soit. Une fois ces
égalités exclues (comme il se doit pour toute comparaison honnête, et
comme c'est indispensable pour comparer équitablement à un modèle non
linéaire qui n'a structurellement aucune raison de tomber pile sur 0),
les chiffres ci-dessus sont les bons. Corrigé dans
`train_pairwise_model.py` et `reference/diagnose_pairwise_metrics.py`.

Constat corrigé : le lien "précision sur paires serrées prédit le
gating" ne tient PLUS aussi proprement qu'annoncé initialement -- k=5 est
maintenant SOUS le hasard (47.9%) sur les paires serrées, pire que
bootstrap gen1 (58.4%) qui pourtant perd plus largement au gating (7/30
contre 9/30). La tranche serrée reste informative (k=1, le meilleur au
gating, est aussi le seul dont aucune tranche n'est sous le hasard), mais
ce n'est pas le prédicteur parfait annoncé initialement -- prudence sur
toute conclusion tirée d'un échantillon de 3 modèles.

## Ré-entraînement pondéré vers les paires serrées : testé, négatif (18/08)

Hypothèse naturelle suite au diagnostic ci-dessus : si R²/MAE surpondère
les gros écarts, reponderer l'entraînement vers les petits `|diff|`
devrait corriger le biais et faire remonter la précision sur les paires
serrées. Implémenté dans `train_pairwise_model.py` (argument optionnel
`tau` : poids = `1/(|diff|+tau)`). Testé sur `pairwise_dataset.npz` (le
dataset k=5), trois façons :

| Approche | Précision de signe, \|diff\|<3 (égalités exactes exclues) |
|---|---|
| Non pondéré (référence) | 47.9% |
| Pondéré, tau ∈ {1.5, 3, 6, 10} | 47.9-48.1% (aucune tendance) |
| Pondéré EXTRÊME (poids ×50 sur les paires serrées) | 47.4% |
| Entraîné **exclusivement** sur les paires serrées (bypass total de la pondération) | 47.3% |

(Chiffres corrigés le même jour après découverte du bug de comptage des
égalités exactes décrit plus haut -- la conclusion qualitative ne change
pas, mais elle est encore plus nette : le modèle linéaire tourne
**au niveau du hasard, voire en dessous**, sur les paires serrées, quelle
que soit la pondération.)

**Négatif, sans ambiguïté.** Même en supprimant purement et simplement
toutes les paires à grand écart de l'entraînement (le cas extrême, où le
modèle ne voit plus JAMAIS un exemple qui pourrait le distraire vers les
gros écarts), la précision sur paires serrées ne bouge pas d'un point.
Ça élimine la fonction de perte comme cause : ce n'est pas que le
compromis R²/MAE "vole" de la capacité aux paires serrées au profit des
grosses -- c'est que le modèle linéaire sur ces 78 features **ne peut
tout simplement pas** discriminer les paires serrées, quelle que soit la
façon dont on pondère ou filtre les données d'entraînement. Plafond de
capacité du modèle (ou de ses features), pas un problème d'optimisation
-- voir la section suivante, qui teste directement si un modèle non
linéaire fait mieux sur ce point précis.

Pas de nouveau gating lancé pour cette tentative : les métriques offline
ne bougeant pas de façon significative par rapport au modèle k=5 déjà
gaté (9/30), relancer un tête-à-tête de 30 parties (~4 min) sur un modèle
statistiquement indiscernable n'aurait rien appris de plus.

**Bilan à ce stade** : le vrai facteur limitant identifié par cette
session (18/08) n'est ni la distribution d'entraînement (bootstrap,
testé négatif) ni la fonction de perte (pondération, testée négative) --
c'est la capacité du **modèle linéaire** à discriminer des paires de
coups presque équivalentes (sur ces mêmes 78 features). Reste à savoir si
c'est un plafond de MODÈLE (un non-linéaire ferait mieux sur les mêmes
features) ou de FEATURES (aucun modèle ne peut faire mieux avec cette
information). Voir la section suivante.

## Un modèle non linéaire (Gradient Boosting) discrimine mieux les paires serrées (18/08)

Suite logique : la tentative MLP pairwise précédente (voir plus haut) est
non linéaire mais a échoué -- ça pourrait clore la question "un modèle
plus expressif aide-t-il ?". Mais cette tentative n'a jamais été évaluée
avec la métrique corrigée ci-dessus (elle a été rejetée hors ligne, sur
R²/MAE honnête post-fuite, avant même d'atteindre le stade du gating) --
donc pas de réponse propre à "un non-linéaire bat-il le linéaire sur les
paires serrées, spécifiquement ?". Testé directement, en pur diagnostic
(pas encore branché dans MCTS) : `HistGradientBoostingRegressor` de
sklearn, entraîné sur la même cible `Xd -> yd` que le modèle linéaire,
même split. Validation croisée à 4 blocs groupés par partie (pas juste un
split unique), 6 configurations d'hyperparamètres (profondeur 2-6, taux
d'apprentissage 0.03-0.05, L2 1-3) :

| Modèle | Précision de signe CV, \|diff\|<3 (égalités exclues) |
|---|---|
| Linéaire (Ridge) | 47.7% ± 0.6% (sous le hasard) |
| Gradient Boosting (HGB), 6 configs testées | 57.3-57.7% ± ~1% (stable sur toute la plage) |

**Positif, et robuste** -- gain de ~10 points, insensible aux
hyperparamètres testés (donc pas un coup de chance sur un tirage
particulier). Sur le jeu de test unique (hors CV), le Gradient Boosting
bat aussi le linéaire sur les tranches [3,8) et reste comparable sur les
tranches larges (déjà faciles pour tout modèle). Contrairement à
l'hypothèse envisagée après le rejet du MLP ("c'est un plafond de modèle,
pas de fonction de perte, donc rien à attendre de plus"), il y a bien de
la structure non linéaire exploitable dans ces 78 features que le modèle
linéaire ne capte pas -- juste pas via l'architecture (MLP siamois) ni
le protocole d'entraînement testés jusqu'ici.

**Piège à éviter avant de brancher ça dans MCTS** : ce diagnostic entraîne
`HistGradientBoostingRegressor` directement sur `Xd -> yd` (comme le
modèle linéaire), ce qui donne un prédicteur de DIFFÉRENCE valide pour
comparer deux candidats au même nœud, mais **ne se décompose pas** en
fonction de valeur par état (`value(état) = w.feat(état)` marche
seulement parce que le modèle est linéaire ; `arbre(a) - arbre(b) !=
arbre(a-b)` pour un ensemble d'arbres). `leaf_eval(state) -> valeurs`
attend une fonction par état, pas une fonction de paire -- brancher ce
modèle tel quel demanderait soit un modèle de valeur ABSOLU par état
(déjà tenté en MLP, négatif), soit une architecture siamoise (déjà
tentée en MLP, négatif après correction de fuite). Piste la plus
prometteuse pour éviter ce piège : utiliser le comparateur pairwise
DIRECTEMENT là où `gen_pairwise_dataset.py` échantillonne déjà ses
candidats -- au même nœud, dans `greedy_action` et dans le rollout de
`search.py` -- comme règle de décision ("lequel des candidats est
meilleur ?") plutôt que comme `leaf_eval` MCTS. Pas encore implémenté ;
proposé à Mehdi avant d'investir dans cette intégration.

## Comparateur pairwise dans greedy_action : implémenté, positif (18/08)

Suite du diagnostic ci-dessus, comme proposé. `search.greedy_action`
accepte désormais un paramètre optionnel `tiebreak(candidate_states,
reference_state, observer) -> list[float]` (opt-in, `None` par défaut --
comportement inchangé si non fourni). Intégration choisie pour éviter le
piège de décomposabilité : le delta exact reste le classement PRINCIPAL
des candidats (inchangé) ; seuls les candidats à moins de
`tiebreak_margin` (3 pts par défaut, la tranche validée par
`diagnose_nonlinear_capacity.py`) du meilleur gain exact sont départagés
par le modèle, TOUS comparés en une seule fois à ce meilleur (comparaison
"en étoile", voir plus bas pour pourquoi). `reference/value_policy.py`
fournit `make_pairwise_gbm_tiebreak()`, qui charge
`pairwise_gbm_model.joblib` (entraîné par `train_pairwise_gbm.py`, voir
plus haut).

Gaté contre B (`reference/gate_pairwise_tiebreak.py`, greedy_action avec
tiebreak vs greedy_action sans, sièges alternés, n=100) :

**E (greedy + tiebreak GBM) vs B : 74/100, écart moyen +50.1 (SE 10,6),
médiane +50,0.**

**Positif, net, et le premier résultat de cette investigation à
effectivement battre B.** ~4.7 écarts-types au-dessus de zéro -- pas un
bruit statistique. Confirme que le signal détecté hors ligne (Gradient
Boosting +10 pts de précision de signe sur les paires serrées, validé en
CV) se traduit bien en victoires réelles une fois branché correctement
(en évitant le piège `leaf_eval`/décomposabilité qui avait fait échouer
le MLP). **E est, à ce jour, le bot le plus fort du dépôt** -- toutes
les mentions antérieures de "B reste le meilleur bot du dépôt" dans ce
fichier sont datées (avant le 18/08 après-midi) et doivent être lues
comme telles.

### Regroupement des appels au modèle (même jour, suite)

Mesure du coût initial (tournoi séquentiel, un appel `model.predict` par
candidat proche) : partie complète gloutonne SANS tiebreak ~18 ms
(`bench.py`) ; AVEC tiebreak sur un seul des deux joueurs, ~2,6 s --
environ ×140. Profilé avant d'incriminer l'algorithme : le clonage de
partie (`Game.clone()+apply()`, ~15 µs) et le calcul des features
(~40 µs) sont négligeables -- **le coût vient à ~99% de l'appel
`model.predict()` lui-même, ~2,4 ms**, un overhead fixe de sklearn par
appel (validation d'entrée, dispatch), pas de la complexité de
l'algorithme. Vérifié que cet overhead est quasiment fixe, pas
proportionnel au nombre de lignes : prédire un batch de 50 lignes en un
seul appel coûte ~2,3 ms au total (45 µs/ligne) contre ~2,4 ms pour 1
seule ligne -- un facteur ~50 par ligne en regroupant.

**Corrigé** : `tiebreak` est passé d'un tournoi séquentiel (un appel par
candidat, mise à jour du "meilleur courant" au fil de l'eau) à une
comparaison en étoile (tous les candidats proches comparés EN UN SEUL
APPEL au meilleur trouvé par le delta exact, `make_pairwise_gbm_tiebreak`
mis à jour en conséquence dans `reference/value_policy.py`). Semantique
légèrement différente (étoile contre un ancrage fixe, plutôt qu'un
tournoi qui pourrait changer d'ancrage en cours de route) mais
mathématiquement plus proche de comment le modèle a été entraîné (paires
depuis un même nœud), et re-gaté pour vérifier l'absence de régression :

**E (batché) vs B : 76/100 (2 nuls), écart moyen +66,0 (SE 10,3),
médiane +72,5** -- au moins aussi bon que la version séquentielle (dans
le bruit), et le gating complet (100 parties) est passé de 106 s à 24 s.
Partie complète avec tiebreak sur un seul joueur : **0,34 s** (contre
2,6 s), soit **~×7,6** au lieu de ×140 par rapport au greedy nu -- pas
encore négligeable, mais un tournoi typique n'a que 2-3 candidats proches
à la fois (pas 50), donc l'amortissement réel est plus modeste que le
×50 mesuré sur un batch synthétique. Reste trop cher pour un rollout MCTS
en l'état (`greedy_action` y est appelé des dizaines de fois par
simulation, des centaines de simulations par décision), mais nettement
plus proche d'être viable qu'avant ce correctif.

Pas encore promu comme comportement par défaut de `greedy_action` :
`tiebreak` reste opt-in (`None` par défaut) pour ne pas forcer une
dépendance à `sklearn`/`joblib`/`reference/` dans `search.py`, qui reste
volontairement libre de ces dépendances (cohérent avec le design déjà en
place pour `leaf_eval` de MCTS, fourni par `reference/value_policy.py`
plutôt que câblé en dur).

## Tiebreak branché dans MCTS : gagne contre l'ancien MCTS, pas contre greedy+tiebreak (18/08)

Suite naturelle : maintenant que le tiebreak batché est bon marché
(~2-3 ms), peut-il aider MCTS lui-même, pas seulement `greedy_action` ?
Piège à éviter, vérifié avant d'intégrer : sur un échantillon réel, **82%
des tours à plusieurs candidats déclenchent un "presque à égalité"** côté
`greedy_action` -- brancher le modèle DANS le rollout (des dizaines de
coups simulés par itération, des centaines d'itérations par décision)
multiplierait le temps par décision par plusieurs dizaines. Intégration
retenue à la place : `MCTS.choose()` accepte un `tiebreak` optionnel
(même signature batchée) utilisé **une seule fois**, à la toute fin,
pour départager les enfants de la racine dont le nombre de visites est
proche du meilleur trouvé (`tiebreak_margin`, 20% du max par défaut) --
coût mesuré négligeable (0,1276s vs 0,1285s sur un test à 150 itérations).

Gaté (`reference/gate_mcts_tiebreak.py`, MCTS(150it) + tiebreak final,
n=30) contre deux adversaires :

| Adversaire | Résultat | Écart moyen (SE) | Lecture |
|---|---|---|---|
| D (MCTS actuel, sans tiebreak) | 18/30 | +45,9 (20,8) | positif, ~2,2 écarts-types -- signal réel mais faible |
| E (greedy + tiebreak, sans MCTS) | 17/30 (1 nul) | +8,5 (22,1) | ~0,4 écart-type -- **statistiquement nul** |

**Conclusion : brancher le tiebreak dans MCTS améliore MCTS par rapport
à lui-même, mais le résultat obtenu ne dépasse pas ce que `greedy_action`
+ tiebreak fait déjà tout seul, sans arbre de recherche.** Cohérent avec
le motif déjà documenté plus haut dans ce fichier (tournoi B vs D,
16/08) : sur ce jeu, un greedy bien réglé résiste étonnamment bien à
l'ajout d'une recherche arborescente, et ce nouveau résultat confirme que
ça reste vrai même une fois le greedy ET le MCTS tous deux équipés du
même comparateur. Le gain de MCTS+tiebreak sur MCTS seul (D) semble
essentiellement venir du tiebreak lui-même, pas d'une synergie avec la
recherche.

**Pas de nouveau meilleur bot ici.** E (greedy + heuristiques + tiebreak,
sans MCTS) reste la référence pratique : aussi fort que F (MCTS +
tiebreak) sur cette mesure, mais très largement moins cher (pas de
recherche arborescente à faire tourner). F reste une alternative valide
(pas rejetée -- le signal contre D est réel), mais rien ne justifie de
le préférer à E en l'état. Code gardé (`search.MCTS(..., tiebreak=...)`)
pour quiconque voudrait creuser plus loin (n plus grand contre E,
`tiebreak_margin` différent, etc.), mais pas de suite priorisée par
défaut.

## Prime de pose d'arbre pilotée par le déficit de slots libres, adoptée (18/08 soir)

Question de Mehdi : la prime de pose d'arbre (`tree_bonus / (1 + n_trees)`,
une décroissance générique) a-t-elle un sens plus proche des règles du
jeu ? Un arbre vide consomme une pose sans rapporter de point -- la
prime ne devrait s'activer que si les slots libres actuels ne suffisent
pas au besoin de placement à venir, pas simplement décroître avec le
nombre d'arbres déjà plantés. Mesure préalable (30 parties MCTS, config
recommandée) : **taux de remplissage moyen 41,2%, 25,9% des arbres
totalement vides (0/4)** -- une part significative des poses d'arbres
n'est jamais rentabilisée, cohérent avec l'intuition.

Prototype (`reference/slot_aware_tree_bonus_experiment.py`) : `slots_libres
= 4*n_trees - slots_occupés`, `besoin = nombre d'habitants en main` (seul
proxy légal du besoin à venir -- main adverse et pioche future cachées),
`déficit = max(0, besoin - slots_libres)`, prime = `tree_bonus *
min(1, déficit/need_scale)`. Pas de plancher : si les slots libres
couvrent déjà la main, planter un arbre ne rapporte plus que son
`delta_tree` réel (~0).

Gaté deux fois, seul le terme de pose d'arbre change (reste identique
sinon) :

| Adversaire | n | Résultat | Écart moyen (SE) | Lecture |
|---|---|---|---|---|
| B (greedy sans tiebreak, référence avant le 18/08) | 2000 | 1059/2000 (53,0%) | +6,2 (2,2) | positif, net, ~2,8 écarts-types |
| E (greedy + tiebreak GBM des deux côtés, bot actuel le plus fort) | 400 | 210/400 (52,5%) | +10,6 (5,4) | positif, ~2 écarts-types -- à la limite de la significativité |

`need_scale` testé de 1 à 8 : peu sensible dans la plage 2-8 (la formule
sature -- le déficit observé est presque toujours soit nul soit déjà bien
au-dessus du seuil), plus faible à 1. Retenu : `need_scale=3.0` (le
réglage le mieux échantillonné, 400 parties contre E).

**Jamais négatif sur aucune configuration testée, mais le signal contre E
est plus faible qu'espéré** -- probablement parce que le tiebreak GBM
d'E corrige déjà une partie des mauvaises poses d'arbres via son propre
signal appris sur des parties réelles, réduisant la valeur ajoutée d'une
heuristique de pose plus juste en amont. **Adopté comme nouveau défaut de
`search.greedy_action`** malgré ce signal modeste : la direction est
constante sur toutes les configs testées, et le principe (ne pas payer de
prime quand les slots suffisent déjà) est une déduction directe des
règles plutôt qu'une décroissance arbitraire -- dans l'esprit de
`reference/MODELS.md` ci-dessus, un résultat positif mais faible reste un
résultat, pas une raison de ne pas l'adopter quand il ne coûte rien
(même complexité, même coût d'exécution) et ne régresse jamais. Un
échantillon plus large contre E (1000-2000 parties) resterait utile pour
resserrer l'intervalle avant de considérer la question tranchée.

Prochaine piste évoquée par Mehdi, non commencée : E reste blindé
d'heuristiques (Clairière, urgence, tiebreak, prime de pose) ; un bot
qui n'utiliserait QUE la déduction pure des règles du jeu (sans aucune
prime ajustée à la main) et battrait quand même E serait un résultat
nettement plus intéressant qu'une nouvelle heuristique de plus -- mais
demande une approche différente (recherche plus profonde ? modèle appris
end-to-end ?), pas encore explorée.

### Le taux de remplissage final ne bouge presque pas -- pourquoi

Question de Mehdi, suite logique : les deux heuristiques (prime de pose
d'arbre, remplissage des slots) jouent-elles vraiment l'une contre
l'autre ? Mesuré en self-play greedy pur (pas MCTS, mêmes graines des
deux côtés, ancienne vs nouvelle formule, 2000 parties x2 = 4000 forêts
par formule) :

| | Ancienne formule | Nouvelle formule |
|---|---|---|
| Arbres/forêt | 22.26 | 22.14 |
| Taux de remplissage | 37.1% (SE 0.17) | 37.5% (SE 0.17) |
| Arbres vides (0/4) | 46.7% | 46.6% |
| Arbres pleins (4/4) | 24.8% | 25.4% |

**Quasiment aucun effet sur le résultat final** -- ~0.4 pt de
remplissage, à peine 1.5-2 écarts-types, alors que le gain de score
mesuré plus haut est net. Diagnostic plus fin : noter l'état (slots
libres, habitants en main) juste AVANT chaque pose d'arbre réelle,
groupé par rang de pose du joueur (1er-3e arbre, 4e-8e, 9e+) :

| Rang de pose | Ancienne (poses "justifiées", déficit>0) | Nouvelle |
|---|---|---|
| 1-3 (ouverture) | 48.8% (n=6000) | **53.8%** |
| 4-8 (milieu) | 1.3% (n=10000) | 2.1% |
| 9+ (fin) | 0.0% (n=28000+) | 0.0% |

**Le mécanisme existe bien, mais concentré presque exclusivement à
l'ouverture** (+5 pts de poses "justifiées" sur les 3 premiers arbres).
Il s'écroule après le 4e arbre parce que le déficit moyen devient déjà
massivement négatif (-10 puis -39 en fin de partie, jusqu'à 28 slots
libres en moyenne avant une pose) -- un excédent structurel créé par le
repli forcé de fin de main : quand `best_gain <= 0` et la main déborde
(> 7 cartes), `greedy_action` doit jouer le meilleur candidat même s'il
est nul, souvent un arbre inutile faute d'alternative. Cet excédent,
une fois créé, pollue le calcul de déficit pour TOUTES les poses
suivantes de la partie, quelle que soit la formule -- donc les deux
formules convergent après coup, peu importe laquelle a décidé plus
finement au début.

**Conclusion** : le gain de score vient très probablement du meilleur
timing des 2-3 premiers arbres, pas d'une réduction globale du gâchis.
Le vrai facteur structurel du taux de remplissage observé (37-41%
suivant la politique) est ce repli forcé de fin de main, pas la formule
de prime -- piste suivante suggérée par Mehdi pour continuer dans cette
direction plutôt que re-régler encore la prime de pose.

### Correction : le repli forcé n'est pas le moteur -- la vraie cause (18/08, suite)

Le repli forcé (`best_gain <= 0 et main > 7 cartes`) évoqué juste
au-dessus comme explication de l'excédent de slots est un **faux-piste**,
vérifié directement : il ne se déclenche que dans **0.2% des décisions**
(116 sur 61745, mesuré sur 300 parties). Il ne peut pas expliquer un
excédent de 22 arbres/forêt en moyenne.

Diagnostic plus direct : journaliser, pour chaque pose d'arbre RÉELLEMENT
choisie (chemin normal, pas le repli), la valeur de `slot_bonus` et de
`delta_tree` séparément (300 parties, 13069 poses observées) :

- **92.9% des poses d'arbre ont `slot_bonus == 0`** (déficit nul -- la
  prime ne contribue rien à cette décision précise)
- Et l'arbre gagne quand même, parce que **`delta_tree` seul vaut en
  moyenne +7.16 points, positif dans 99.6% des cas**
- Seulement **7.1%** des poses d'arbre s'appuient réellement sur la
  prime pour l'emporter

**La prémisse de tout ce module ("un arbre ne rapporte presque rien à la
pose", docstring de `greedy_action` depuis l'origine) est fausse dans
l'immense majorité des décisions réelles.** Elle n'est vraie qu'à
l'ouverture, avant que des habitants déjà en jeu n'aient un score qui
dépend du nombre d'arbres/de la diversité d'espèces (majorités,
comptages type "x arbres", etc.) -- une fois quelques-uns de ces
habitants posés, chaque nouvel arbre revalorise rétroactivement ce qui
est déjà en forêt, et `delta_tree` devient substantiellement positif
tout seul, sans avoir besoin d'aucune prime artificielle.

**Ça explique proprement les deux résultats précédents** :
- Pourquoi ancienne/nouvelle formule de prime changent si peu le
  remplissage final : 93% des poses n'utilisent la prime pour rien,
  `delta_tree` décide seul.
- Pourquoi l'effet mesuré de la nouvelle formule se concentre sur les 3
  premiers arbres : c'est exactement les ~7% de poses où `delta_tree`
  ne suffit pas encore.

**Corrigé** : la docstring de `search.greedy_action` reformulée pour ne
plus affirmer que le delta est "quasi nul" en général -- ce n'est vrai
qu'en ouverture (voir `search.py`, commit du 18/08 soir).

### Piste ouverture : geler la prime au-delà d'un seuil d'arbres -- neutre, vérifié (18/08)

Suite logique proposée par Mehdi : si la prime ne sert qu'à l'ouverture,
la geler à 0 dès que `n_trees` dépasse un seuil (au lieu de calculer le
déficit à chaque pose) devrait être neutre en force. Vérifié directement
sur des parties réellement jouées (sièges alternés, mêmes graines) :

| Seuil (`n_trees <`) | Adversaire | n | Résultat |
|---|---|---|---|
| 2 | défaut actuel | 500 | 254/500 (50.8%), -4.9 (SE 4.2) |
| 4 | défaut actuel | 1000 | 505/1000 (50.5%), -2.3 (SE 3.2) |
| 4 | E (tiebreak GBM des deux côtés) | 300 | 153/300 (51.0%), +1.2 (SE 6.1) |
| 6 | défaut actuel | 500 | 264/500 (52.8%), -0.7 (SE 4.2) |
| 8 | défaut actuel | 500 | 264/500 (52.8%), -0.7 (SE 4.2) |

**Confirme sans ambiguïté, sur des parties jouées et pas seulement en
diagnostic statique, que la prime est inerte au-delà de l'ouverture.**
Geler le calcul est une simplification neutre en force (jamais un gain
ni une perte mesurable sur aucune config testée) -- pas encore adoptée
par défaut dans `search.py` : le coût du calcul en trop (quelques
additions par décision) est négligeable face au tiebreak GBM ou à MCTS,
donc l'intérêt est surtout documentaire (le code refléterait alors
explicitement ce que le comportement mesuré montre déjà), pas une
optimisation qui vaille la peine seule. À adopter si `search.py` est
retouché pour d'autres raisons dans cette zone, sinon laissé tel quel.

## Décomposition du score final par mécanisme -- où sont les gros leviers (18/08)

Question de Mehdi, suite naturelle : si planter des arbres rapporte des
points concrets (~7pts/arbre en moyenne, voir plus haut), quels sont les
GROS leviers de score en général -- additif brut, multiplicatif
(cervidés, lapin/renard...), séries à paliers (Marronnier, papillons...),
seuils tout-ou-rien (Lynx, chauve-souris...) ?

Outil (`reference/gen_score_breakdown.py`) : copie carte-à-carte de
`scoring_ref.score_forest` (l'oracle de scoring littéral, une règle =
une fonction) qui accumule les points par carte au lieu de les sommer --
appliqué aux forêts finales de E (greedy + tiebreak GBM) en self-play,
300 parties (600 forêts), sièges alternés.

**Cartes individuelles les plus lourdes** (pts/forêt, % du score total) :

| Carte | pts/forêt | % du score |
|---|---|---|
| ROE_DEER (Chevreuil) | 54.4 | 11.7% |
| SYCAMORE (Sycomore) | 48.7 | 10.5% |
| BEECH_MARTEN (Fouine) | 39.6 | 8.5% |
| RED_DEER (Cerf) | 32.3 | 6.9% |
| TREE_FERNS (Fougère arborescente) | 23.2 | 5.0% |
| WOOD_ANT (Fourmi des bois) | 21.5 | 4.6% |
| HORSE_CHESTNUT (Marronnier, set) | 20.5 | 4.4% |
| OAK (Chêne, seuil 8 espèces) | 20.2 | 4.4% |
| FALLOW_DEER (Daim) | 19.2 | 4.1% |

**Regroupé par mécanisme** :

| Levier | pts/forêt | % du score | Cartes |
|---|---|---|---|
| **Cervidés (multiplicatif)** | **117.8** | **25.3%** | ROE_DEER (×symboles de l'arbre porteur), RED_DEER (×Arbre+Plante), FALLOW_DEER (×ClovenhoofedAnimal), WOLF (×Cervidés) |
| Multiplicatif "autres types" | 84.6 | 18.2% | TREE_FERNS (×Amphibien), WOOD_ANT (×slots Bottom occupés), GOSHAWK (×Oiseau), BULLFINCH (×Insecte), BLACKBERRIES (×Plante), HEDGEHOG (×Papillon), STAG_BEETLE (×PawedAnimal), GNAT (×Bat), TREE_FROG (×Moustique/GNAT) |
| Arbres multiplicatifs | 58.5 | 12.6% | SYCAMORE (×nb d'arbres), SILVER_FIR (×slots occupés sur lui-même) |
| Arbres à seuil | 49.9 | 10.7% | OAK (10pts si ≥8 espèces), BEECH (5pts si ≥4 Hêtres), LINDEN (majorité), MOSS (10pts si ≥10 symboles arbre) |
| **Fouine seule (BEECH_MARTEN)** | **39.6** | **8.5%** | 5pts × nb d'arbres à 4/4 slots -- une seule carte, quasi aussi lourde qu'une catégorie entière |
| Séries à paliers | 37.7 | 8.1% | HORSE_CHESTNUT, FIRE_SALAMANDER, FIREFLIES, BUTTERFLY_SETS, chauves-souris (5pts/carte si ≥3 espèces) |
| Seuil binaire divers | 38.6 | 8.3% | LYNX (si Chevreuil présent), WILD_BOAR (si Loir gris... SQUEAKER), WILD_STRAWBERRIES, GREAT_SPOTTED_WOODPECKER (majorité), COMMON_TOAD, EUROPEAN_FAT_DORMOUSE, CHAFFINCH, RED_SQUIRREL |
| Additif fixe | 24.5 | 5.3% | DOUGLAS_FIR, BIRCH, TAWNY_OWL, POND_TURTLE, EURASIAN_JAY, EUROPEAN_BADGER, SQUEAKER, Grotte |
| Lapin / renard | 13.5 | 2.9% | EUROPEAN_HARE (×lièvres), RED_FOX (2×lièvres) |

**Le levier dominant, de loin : les cervidés (25.3% du score à eux
quatre), avec ROE_DEER seul déjà la carte la plus lourde du jeu
(11.7%).** Confirme et affine le résultat précédent sur `delta_tree` :
les arbres ne sont pas juste un levier "de base" séparé -- ils sont le
SUBSTRAT qui alimente la plupart des gros leviers. ROE_DEER compte les
**symboles d'arbre** (chaque arbre + chaque habitant qu'il porte), pas
les arbres physiques seuls, donc planter massivement nourrit ce levier
en plus des catégories "arbres" listées à part (58.5 + 49.9 = 108.4
pts/forêt, 23.3%, rien que pour les deux catégories explicitement
arborescentes) -- la vraie contribution des arbres au score, en comptant
leur effet indirect via ROE_DEER/WOOD_ANT/MOSS, est donc encore plus
large que ces deux lignes seules.

**BEECH_MARTEN (Fouine) mérite un coup d'œil à part** : une seule carte
pèse 8.5% du score total, presque autant qu'une catégorie entière à elle
seule -- prochaine piste demandée par Mehdi.
