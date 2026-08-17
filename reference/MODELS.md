# Modèles de valeur — quel fichier est quoi

Plusieurs `.joblib`/`.npz` cohabitent dans `reference/` parce que chaque
tentative d'amélioration du modèle de valeur a été gardée comme point de
comparaison plutôt que supprimée (voir la philosophie générale du dépôt :
un résultat nul ou négatif reste un résultat, pas une erreur à effacer).
Ce fichier sert d'index pour ne pas s'y perdre.

## Fichiers vivants (chargés par défaut)

| Fichier | Rôle |
|---|---|
| `pairwise_model.joblib` | Modèle de valeur contrastif **utilisé par défaut** par `value_policy.make_pairwise_hybrid_leaf_eval()` (donc par tout MCTS instancié sans `model_path` explicite). Réentraîné le 16/08 (2e fois) avec `gen_pairwise_dataset.py` + `train_pairwise_model.py` sous le code actuel : heuristique `choose_draw_source` **forte** + urgence Clairière (`CLEARING_URGENCY_BONUS`) dans `greedy_action`. |
| `pairwise_dataset.npz` | Dataset de paires (diff features -> diff gain réel) ayant servi à entraîner `pairwise_model.joblib` ci-dessus. 150 parties, 21468 paires, 78 features. |
| `value_model.joblib`, `value_dataset.npz` | Modèle de valeur **absolu** (MLP), approche antérieure au modèle contrastif. Toujours chargé par `value_policy.make_leaf_eval()`/`make_hybrid_leaf_eval()`, utilisé par `bench.py` comme point de comparaison. Limite connue : MAE (~12 pts) plus grande que l'écart réel entre deux coups candidats (~9 pts) -- voir la docstring de `gen_pairwise_dataset.py`, c'est justement pour corriger ce défaut que l'approche contrastive a été introduite. |

## Sauvegardes historiques (non chargées par défaut, gardées pour comparaison/rollback)

| Fichier | Contexte |
|---|---|
| `pairwise_model_pre_clairiere_urgence.joblib`, `pairwise_dataset_pre_clairiere_urgence.npz` | Le modèle tel qu'il était **avant** l'ajout de l'urgence Clairière (`CLEARING_URGENCY_BONUS`) à `greedy_action` -- entraîné sur des trajectoires générées sans cette urgence. Réutiliser ce modèle avec le `greedy_action` actuel recrée le même problème de distribution périmée : classement agrégé de D tombé de 63.3% à 51.1% dans le tournoi (voir plus bas), corrigé par le réentraînement suivant. |
| `pairwise_model_pre_clairiere_forte.joblib`, `pairwise_dataset_pre_clairiere_forte.npz` | Le modèle contrastif tel qu'il était **avant** le premier réentraînement du 16/08 -- entraîné sous l'ancienne heuristique de Clairière (toujours la carte la moins chère). C'est ce modèle périmé qui faisait perdre MCTS+forte face à Greedy+forte (12/30) ; le réentraînement (voir `bench_heuristics.py`) fait remonter le taux de victoire à ~59%. Gardé pour pouvoir reproduire la comparaison avant/après. |
| `pairwise_model_stale_backup.joblib`, `pairwise_dataset_stale_backup.npz` | Snapshot du 15/08 19h48, avant le réentraînement "sous les règles actuelles" de cette même journée (commit `233b14e`) -- prédate même l'ajout des features Clairière (`clearing_size`/`clearing_min_cost`) au vecteur de features. Conservé comme point de repère le plus ancien. |
| `pairwise_model_true_original_aligned.joblib`, `pairwise_model_retrain1_no_clearing_aligned.joblib` | Paire de modèles produits lors de l'expérience "ajouter les features Clairière au modèle de valeur" (commit `013edbc`, 15/08 19h57) : le premier reprend l'entraînement d'origine réaligné sur le nouveau schéma de features (colonnes Clairière à zéro) pour servir de témoin, le second est réentraîné avec les features Clairière réellement peuplées. Résultat **nul** (l'ajout de ces features n'a pas amélioré le jeu de façon mesurable) -- gardés comme trace de la tentative, pas comme candidats à adopter. |
| `bootstrap_dataset_pilot1_50it.npz`, `bootstrap_model_pilot1_50it.joblib` | Pilote 1 du bootstrap MCTS façon AlphaZero (17/08, voir "Résultat du pilote 1" plus bas) : auto-jeu MCTS 50 it. + cible = stats de l'arbre, au lieu d'auto-jeu greedy + rollout séparé. Résultat **négatif** au gating (perd contre le modèle officiel ET contre B) -- gardés comme trace, `pairwise_model.joblib` reste inchangé. |
| `bootstrap_dataset.npz`, `bootstrap_model.joblib` | Pilote 2, même méthode que le pilote 1 mais budget MCTS plus élevé -- voir "Résultat du pilote 2" plus bas pour le statut courant. |

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

## Piste future (non implémentée) : bootstrap MCTS façon AlphaZero

Documentée ici sur demande de Mehdi (16/08) -- explique la logique, les
étapes et le pourquoi, mais **rien de ce qui suit n'est codé**. C'est un
plan, pas un résultat.

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

## Critère de gating : deux paliers, pas un seul (17/08, précision de Mehdi)

Le plan ci-dessus (étape 4) disait "ne remplacer le modèle officiel que
si la nouvelle génération bat mesurablement l'ancienne ET B en
tête-à-tête" -- formulation trop stricte pour une boucle censée
s'améliorer génération après génération façon AlphaGo Zero. Dans
AlphaGo Zero, le gating par génération ne compare le candidat qu'au
**meilleur réseau courant** (O ici) ; battre une heuristique externe (B)
n'est jamais un péage par génération, sous peine de ne jamais pouvoir
démarrer la boucle. Deux paliers distincts, donc :

- **Bat O** -> le candidat devient la nouvelle base d'auto-jeu pour la
  génération suivante (la boucle avance), même s'il perd encore contre B.
- **Bat O ET B** -> seuil de promotion en modèle "officiel"
  (`reference/pairwise_model.joblib`, celui chargé par défaut) -- la
  ligne d'arrivée, pas un prérequis à chaque étape intermédiaire.

Tant qu'un candidat ne bat pas O, il n'y a pas de génération suivante à
proprement parler : l'auto-jeu recommence depuis O avec un budget
différent, ce n'est pas encore une chaîne bootstrap qui s'auto-améliore.

## Résultat du pilote 1, 50 it. MCTS (17/08) : négatif, gardé pour comparaison

La piste ci-dessus a été codée et testée à l'échelle "pilote" recommandée
(commencer petit). Trois scripts, réutilisant l'infrastructure existante
plutôt que de la dupliquer :

- `gen_bootstrap_dataset.py` : auto-jeu avec un `search.MCTS` persistant
  par joueur (arbre réutilisé d'un tour à l'autre, `leaf_eval` = modèle
  officiel `pairwise_model.joblib` via `make_pairwise_hybrid_leaf_eval`),
  au lieu de la trajectoire `S.greedy_action` de `gen_pairwise_dataset.py`.
  Cible d'entraînement = `child.value / child.visits` des actions
  candidates au nœud racine (option 2 du plan ci-dessus, stats de l'arbre
  plutôt qu'un rollout séparé), filtrées à `min_visits >= 3`.
- `train_bootstrap_model.py` : réutilise `train_pairwise_model.fit_and_save`
  (factorisée pour l'occasion, comportement par défaut inchangé) sur le
  dataset bootstrap.
- `bench_bootstrap.py` : gating tête-à-tête, même format que
  `bench_heuristics.py` -- N (candidat bootstrap) vs O (officiel) vs B
  (Greedy + Clairière forte, meilleur bot du dépôt).

**Génération** : 30 parties, MCTS 50 itérations/décision -> 19841 paires
en 107s (`reference/bootstrap_dataset_pilot1_50it.npz`).

**Entraînement** (`reference/bootstrap_model_pilot1_50it.joblib`) : R² = 0,037, MAE en
rang normalisé [0,1] = 0,14, **pire que la baseline "aucune différence"
(0,13)**. Signal offline aussi faible que la toute première tentative de
modèle contrastif (R²=0,045, sans le score exact en feature), bien en
dessous du modèle officiel actuel (R²=0,17, `k_rollout=5`).

**Gating** (`bench_bootstrap.py --n 20`, MCTS 150 it., même heuristique de
Clairière forte des deux côtés) :

| Match | Score | Écart moyen (SE) |
|---|---|---|
| N vs O | 5/20 | -3,0 (12,4) |
| N vs B | 8/20 | -43,7 (24,2) |
| O vs B | 11/20 | -23,6 (26,7) |

**Le candidat N perd nettement contre le modèle officiel O en taux de
victoire (5/20, ~25%, malgré un écart moyen non significatif) et contre B
(8/20). Ne franchit même pas le premier palier (bat O) : pas de
génération suivante possible depuis ce candidat.**
`bootstrap_dataset_pilot1_50it.npz`/`bootstrap_model_pilot1_50it.joblib`
gardés comme trace de la tentative (même philosophie que le reste de ce
fichier), pas comme candidats. Voir plus bas pour le pilote 2 (budget
d'itérations plus élevé, même diagnostic testé).

Diagnostic le plus probable : à 50 itérations MCTS réparties sur un
facteur de branchement de plusieurs centaines d'actions (voir README), la
plupart des candidats au nœud racine ne reçoivent que 3 à ~20 visites
(observé en pratique) -- `child.value/child.visits` sur si peu
d'observations est encore très bruité, probablement pas moins bruité que
le rollout unique que `k_rollout` a justement été introduit pour
corriger. Le "pilote minimal" recommandé par le plan ci-dessus semble
trop petit pour que l'avantage théorique de la méthode (cible = jugement
de la recherche complète, pas d'un seul rollout) se matérialise -- il
faudrait soit beaucoup plus d'itérations MCTS par décision (coût
multiplié d'autant), soit un `min_visits` sensiblement plus élevé (moins
de paires exploitables par partie, donc plus de parties nécessaires pour
un dataset de taille comparable). Piste non refermée : rejouer ce pilote
à budget de calcul nettement supérieur avant de la considérer invalidée,
pas juste mal exécutée à cette échelle.
