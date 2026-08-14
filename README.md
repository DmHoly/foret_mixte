# Forêt Mixte : moteur corrigé, simulateur et recherche

Reprise du projet `foret_mixte_jax_project`, avec les règles vérifiées carte
par carte contre le moteur de référence et le code réécrit pour la vitesse.

La référence utilisée est [soldag/forest-shuffle-scoring](https://github.com/soldag/forest-shuffle-scoring),
cloné et exécuté, pas lu de mémoire. Les données de cartes sont **extraites
mécaniquement** de son TypeScript (`tools/gen_cards.py` -> `cards.py`), donc
elles ne peuvent plus diverger par recopie manuelle.

Le fichier Excel n'était pas joint à la conversation, donc la comparaison
demandée avec cette source n'a pas pu être faite. Elle reste utile comme
troisième point de contrôle sur les effectifs du deck.

---

## 1. Ce qui était faux

Vérifié en exécutant le moteur de référence, pas par lecture.

### 1.1 ROE_DEER : mauvaise règle (impact fort)

L'ancien moteur donnait 3 points par **arbre de la même espèce**. La règle
compte toutes les **cartes portant le même symbole d'arbre**, habitants
compris, et le chevreuil se compte lui-même.

```
ancien : 3 x (nombre d'arbres de l'espèce porteuse)
correct: 3 x (arbres de cette espèce + habitants posés sur ces arbres)
```

Reproduit par `tests/test_rules.py::test_roe_deer_*`, valeurs alignées sur
`RoeDeer.test.ts` (9 / 12 / 15 / 18 / 21 points selon le symbole). Sur une
forêt de fin de partie l'écart atteint plusieurs dizaines de points, et le
chevreuil passe d'une carte moyenne à une carte de premier plan.

### 1.2 COMMON_TOAD : facteur 2

Le score est attribué **par crapaud**, pas par arbre : une paire vaut 10, pas
5. Confirmé par `CommonToad.test.ts`.

### 1.3 VIOLET_CARPENTER_BEE : effet non implémenté

L'abeille n'a pas de points propres, mais elle porte un modificateur
`woodyPlantCount` qui fait **compter son arbre porteur une fois de plus**.
Dans le jeu de base cela change le seuil de MOSS (10 arbres) et la majorité
d'arbres du pic épeiche. L'ancien moteur l'ignorait entièrement.

Subtilité reproduite fidèlement : le modificateur s'applique aux comptages par
nom et à `countCards` brut, mais **pas** à `countCardTypes` ni à
`countTreeSymbols`. SYCAMORE ne bénéficie donc pas de l'abeille, MOSS oui.

### 1.4 Positions autorisées trop permissives (7 habitants)

`RACCOON`, `RED_DEER`, `ROE_DEER`, `SQUEAKER`, `VIOLET_CARPENTER_BEE`,
`WILD_BOAR`, `WOLF` étaient déclarés jouables en Left **et** Right sur chaque
symbole. En réalité chaque exemplaire est une variante précise. Exemple :

```
ancien  ROE_DEER : BEECH/BIRCH/HORSE_CHESTNUT/LINDEN/SILVER_FIR en Left et Right
correct ROE_DEER : LINDEN Left, SILVER_FIR Left, BEECH Right, BIRCH Right,
                   HORSE_CHESTNUT Right
```

Le simulateur générait donc des coups illégaux, ce qui gonfle mécaniquement
les scores et fausse toute comparaison de politiques.

### 1.5 Le deck : 250 cartes au lieu de 158

C'est l'erreur la plus lourde de conséquences pour le simulateur.

```
66 cartes Arbre
48 cartes habitant Top/Bottom   ->  96 moitiés
44 cartes habitant Left/Right   ->  88 moitiés
------------------------------------------------
158 cartes à piocher, portant 250 entités jouables
```

L'ancienne version piochait dans 250 cartes en traitant chaque moitié comme
indépendante. Effets : parties ~60 % plus longues, hiver déclenché trop tard,
grotte sans objet, scores hors échelle.

Les appariements réels des moitiés ne sont pas publiés ; ils sont tirés au
hasard à chaque partie (Top avec Bottom, Left avec Right), ce qui préserve la
taille du deck et les distributions marginales.

### 1.6 Grotte : un point gratuit par habitant

`add_dweller` incrémentait la grotte à chaque pose. Combiné aux majorités
automatiquement gagnées en solo, cela expliquait une bonne part de l'écart
entre les scores simulés (400 à 500) et les scores humains (250 à 320).

Le livret de règles (PDF, à la racine du dépôt) confirme que la Grotte vaut
1 point par carte qu'elle contient en fin de partie, mais ne détaille pas
quelles cartes l'alimentent (renvoi au livret de référence des cartes,
absent du PDF de base). C'est en réalité un effet de carte unique, pas une
règle générique : seul le **Raton laveur** envoie des cartes de la main à la
Grotte (`CAVE_CHOICE_DWELLERS` dans `engine.py`, résolu comme un vrai choix
stratégique via l'action `cave_discard`). Il n'y a pas de point gratuit par
habitant posé ; l'ancien paramètre `cave_per_dweller` a été retiré.

### 1.7 Ce que ma première revue avait signalé à tort

Deux points de `REVUE.md` étaient faux, et la référence les infirme :

- **BROWN_BEAR, MOLE, RACCOON et l'abeille ne sont pas des règles oubliées.**
  Le moteur de référence leur attribue explicitement `score: () => 0`. Ce sont
  des cartes à effet de jeu, comme les champignons. L'ancien moteur avait
  raison. `tests/test_rules.py::test_zero_point_cards_are_intentional` fige ce
  constat pour éviter qu'on ne les "corrige" plus tard.
- **Le plafond de HORSE_CHESTNUT à 7 est correct**, malgré les 11 exemplaires
  du deck. La table de la référence s'arrête bien à 7 (49 points).

Le reste de la revue tient : le point sur `mctx` (pas de rollouts, pas de
support 3+ joueurs) et le point sur le goulot de performance restent valides.

---

## 2. Optimisation

Le scoring ne parcourt plus la forêt. Toutes les quantités dont dépendent les
règles (compteurs par nom, par type, par symbole, arbres pleins, habitants au
Bottom, loirs activés, crapauds appariés, papillons par espèce...) sont
maintenues **incrémentalement** à la pose. `score()` combine une trentaine de
compteurs et boucle sur les 8 espèces d'arbres. Son coût est indépendant du
nombre d'habitants posés.

Deux ajouts qui comptent pour la recherche :

- `undo_dweller` / `undo_tree` : annulation LIFO exacte, donc on évalue un
  coup candidat en posant/scorant/annulant au lieu de copier la forêt.
- `butterfly_score` en forme close : la boucle `while` de constitution de sets
  du moteur d'origine est remplacée par une somme sur les rangs, sans boucle
  dynamique. C'est aussi la forme directement portable en `jnp`.

Mesures sur une forêt de fin de partie (25 arbres, 108 habitants), CPU
mono-cœur :

| | ancien moteur | nouveau | oracle |
|---|---|---|---|
| `score_forest` | 291 us | **4,2 us** | 1 224 us |
| facteur | 1x | **x70** | |

| opération | coût |
|---|---|
| `Forest.score()` | 4,2 us |
| `Forest.copy()` | 15,4 us |
| `Forest.delta_dweller()` | 9,3 us |
| `Game.clone()` | 4,0 us |
| `Game.legal_actions()` | 13,6 us |

Partie complète à 2 joueurs, politique gloutonne : **2,7 ms** contre ~15 s
annoncées pour l'ancienne boucle MCTS-léger.

---

## 3. Réalisme des scores

C'est le contrôle qui compte plus que le chrono.

| | score moyen | maximum |
|---|---|---|
| ancien simulateur (solo) | 400 à 500 | |
| **nouveau, greedy, 2 joueurs** | **212** | **277** |
| bon joueur humain, 2 joueurs (BGA) | 250 à 320 | |

Les parties durent 140 tours au total, soit 70 par joueur. Le simulateur est
maintenant dans le bon régime, ce qui rend les comparaisons de politiques
interprétables. Il ne l'était pas avant.

---

## 4. Recherche

`search.py` contient trois politiques.

**greedy** : maximise le gain de score immédiat, avec une prime décroissante
pour la pose d'arbres (sans elle, une politique gloutonne ne plante jamais
d'arbre et plafonne très bas). 2,7 ms par partie.

**beam** : lookahead en faisceau sur la main. **Moins bon que greedy** ici
(182 contre 212) : chercher profond en ignorant les adversaires et la pioche
sur-optimise des lignes qui ne se réalisent pas. Résultat conservé tel quel,
il est informatif.

**MCTS** : arbre persistant, sélection UCT, information imparfaite.

Quatre choix de conception, et leurs raisons :

1. **Actions identifiées par la carte, pas par l'indice en main.** Un indice
   de main se décale à chaque pioche : la même action désignait des coups
   différents selon la déterminisation, et l'arbre agrégeait des statistiques
   sans rapport. C'est la correction qui a fait passer le MCTS de perdant à
   gagnant contre greedy. Effet secondaire utile : les doublons de main
   fusionnent, le facteur de branchement chute.

2. **Déterminisation par itération** (ISMCTS mono-observateur). Le deck et les
   mains adverses sont rebattus à chaque simulation, les cartes Hiver restant
   dans le deck. Chercher sur l'état réel serait de la triche.

3. **Rollouts tronqués et guidés.** Politique gloutonne avec epsilon, tronquée
   à une profondeur fixe, puis évaluation par le score courant. Le score d'une
   forêt étant monotone croissant, l'évaluation tronquée est peu biaisée. C'est
   l'équivalent bon marché d'une fonction de valeur.

4. **Récompense = rang normalisé**, pas score brut. Le score varie d'un facteur
   3 selon le tirage ; backpropager le brut ferait apprendre la variance du
   tirage plutôt que la qualité des coups.

Résultats contre greedy, sièges alternés, 8 parties (échantillon petit, à
prendre comme une tendance) :

| configuration | victoires | écart moyen | temps/décision |
|---|---|---|---|
| 100 itérations, profondeur 25 | 4/8 | -21,5 | 49 ms |
| 400 itérations, profondeur 40 | **6/8** | **+4,4** | 286 ms |

La progression avec le budget est le signal important : la recherche paie, mais
il faut plusieurs centaines de simulations avant de dépasser une heuristique
gloutonne correcte. C'est cohérent avec un facteur de branchement de plusieurs
centaines d'actions.

### Fonctions de valeur pour `leaf_eval` (session du 14-15/08)

Le rollout tronqué ci-dessus a un coût qui domine le temps par décision
(423 ms à 300 itérations). Objectif : remplacer une partie du rollout par
une évaluation moins chère sans perdre en qualité de décision. Quatre
tentatives, dans l'ordre, avec ce qui a marché et ce qui n'a pas marché :

1. **Modèle absolu (MLP sklearn)** sur des features de composition de
   forêt (proportions par espèce/type, `reference/features.py`,
   `reference/train_value_model.py`). R²=0,93 en isolation, mais **perd
   contre greedy** (2/4, -56,0) une fois branché en `leaf_eval` sans
   rollout : son bruit d'estimation (MAE ~12 pts) dépasse l'écart réel
   entre deux coups candidats à un même nœud (~9 pts), donc MCTS ne peut
   pas s'en servir pour trancher entre branches. Voir
   `reference/value_policy.py::make_leaf_eval`.

2. **Modèle contrastif sans signal de score** (`reference/gen_pairwise_dataset.py`
   v1) : régression sur des DIFFS de features entre coups candidats
   évalués avec la même suite de pioche (common random numbers, pour
   annuler le bruit partagé). R²=0,045 -- quasi aucun signal, les
   features en proportions bougent à peine pour un seul coup.

3. **Modèle contrastif + score brut du candidat en feature** (v2) : le
   score exact du candidat (déjà calculé sans bruit par le moteur) ajouté
   comme feature dans la comparaison de coups soeurs, sans risque de fuite
   d'horloge puisque les candidats comparés partagent le même tour. R²=0,17,
   MAE 6,15 pts contre une baseline "aucune différence" à 6,80 -- signal
   réel mais modeste. En `leaf_eval` pur (`make_pairwise_leaf_eval`,
   produit scalaire, ~0 coût d'inférence) : 5/8 contre greedy, +3,5,
   73 ms/décision -- bat greedy mais nettement moins bien que le rollout.

4. **Rollout court (10 coups réels) + modèle contrastif** (retenue,
   `make_pairwise_hybrid_leaf_eval`) : les coups réels rapprochent l'état
   de la distribution d'entraînement du modèle ET font émerger la
   différence marginale entre branches (que le modèle seul ne voit pas) ;
   le modèle prend le relais moins cher qu'un rollout complet pour le
   reste. Résultat, la meilleure config mesurée à ce jour :

   | configuration | victoires | écart moyen | temps/décision |
   |---|---|---|---|
   | rollout classique (300 it, profondeur 30) | 6/8 | +19,1 | 423 ms |
   | modèle contrastif seul (300 it) | 5/8 | +3,5 | 73 ms |
   | **rollout court + modèle contrastif (300 it)** | **7/8** | **+13,8** | **192 ms** |
   | même config, échantillon large | **17/20** | **+14,3** | -- |

   ~2,2x plus rapide que le rollout classique pour une qualité de jeu
   proche (voire dépassée sur le taux de victoire). C'est la config
   utilisée par `run_narrated_hybrid.py` et `run_stats_hybrid.py`.

### Force intrinsèque des cartes (`reference/card_strength.py`)

Question annexe posée en session : quelle est la "force" d'une carte,
indépendamment de la politique qui l'a jouée ? Mesurée par **retrait
contrefactuel** : rejouer une partie jusqu'au bout, puis retirer chaque
carte posée une par une de la forêt FINALE (reconstruction du journal de
pose, pas `undo_dweller`/`undo_tree` qui sont strictement LIFO) et mesurer
la perte de points. Contrairement au delta immédiat (`greedy_action`), ça
capture la valeur différée (ROE_DEER compte des symboles posés plus tard).

Deux tables générées, greedy (`card_values.py`, 50 parties) et MCTS hybride
(`card_values_mcts.py`, 20 parties, 300 it) :

- **Confirmé** : les cartes gratuites empilables/pairables (Lièvre d'Europe,
  Fourmi des bois, Crapaud commun) dominent le classement, sous les deux
  politiques.
- **Hypothèse infirmée en session** : je pensais l'usage modéré des
  cervidés (ROE_DEER, FALLOW_DEER) dû à un biais d'horizon du MCTS (rollout
  court + modèle entraîné sur ~20 coups). En fait leur `marginal_brut` est
  déjà bon sous greedy (25-26 pts, meilleur que la plupart des cartes à
  coût 1) -- c'est le coût de 2 cartes qui les pénalise arithmétiquement,
  pas un défaut de vision de la recherche. Sous MCTS l'écart est réel mais
  modeste (+2 à +2,5 pts), pas la révélation attendue.
- **Résultat inattendu** : la contribution marginale d'une carte n'est PAS
  purement intrinsèque, elle dépend du contexte stratégique. Sous MCTS
  (forêts plus denses en cartes fortes), le marginal de cartes comme
  WOOD_ANT ou GOSHAWK chute de 8-9 pts par rapport à greedy -- même carte,
  contribution différente selon ce qui l'entoure. Piste non explorée : au
  lieu d'une moyenne plate par carte, une table conditionnée par une
  variable de contexte pertinente (ex. `symbol_count` déjà posé pour
  ROE_DEER, `n_trees` pour MOSS) -- coûterait la même chose à l'usage
  (lookup O(1), tout le travail reste hors-ligne) mais serait un
  découpage choisi à la main plutôt qu'un modèle générique à 76 features
  qui n'a pas trouvé le bon signal (R²=0,17, voir ci-dessus). Pas
  implémenté, à reprendre si on revient sur ce sujet.
- **Tentative d'exploitation ratée** : injecter `marginal_brut` (pondération
  du choix de coup, et/ou paiement par la carte la plus faible plutôt que
  la plus chère en coût facial) dans une politique gloutonne
  (`reference/strength_policy.py`, `reference/bench_strength.py`). Résultat
  au mieux dans le bruit (53/100 à w=0, -0,7 d'écart) et net négatif dès
  que le poids de sélection augmente -- corriger un delta de score exact
  et sans bruit avec une moyenne globale grossière dégrade la décision
  plutôt que de l'améliorer, même constat que pour le MLP en point 1.
  Le greedy actuel (delta exact + `tree_bonus`) résiste bien aux retouches
  heuristiques locales ; le vrai gain vient de la recherche arborescente
  (MCTS), pas d'un greedy affiné.

### Ce qui manque pour aller plus loin

Par ordre de rendement attendu :

1. **Le paiement comme décision de l'arbre.** Aujourd'hui heuristique
   (`choose_payment`). C'est le plus gros trou de modélisation restant, et il
   conditionne les bonus de couleur.
2. **Le choix de la moitié sacrifiée.** Jouer une moitié perd l'autre :
   décision réelle, actuellement prise par « la première qui correspond ».
3. **Parallélisation racine** (multiprocessing) : facteur proche du nombre de
   cœurs, sans changement d'algorithme.
4. **Fonction de valeur apprise.** C'est ce qu'attend `mctx`. Voir `REVUE.md` :
   `mctx` ne fait pas de rollouts, et ne gère nativement que le mono-agent ou
   le 2 joueurs à somme nulle.

Non implémenté côté règles, inchangé depuis la version précédente : pioche
depuis la clairière, vidage de la clairière à 10 cartes, bonus de paiement par
couleur, moteurs de pioche des champignons.

---

## 5. Fichiers

| Fichier | Rôle |
|---|---|
| `cards.py` | Données des 58 cartes. **Généré**, ne pas éditer à la main. |
| `tools/gen_cards.py` | Extraction depuis le dépôt de référence. |
| `tools/base_cards.json` | Sortie brute de l'extraction, pour diff. |
| `scoring_ref.py` | Oracle : transcription littérale du moteur TypeScript. |
| `engine.py` | Moteur optimisé : compteurs incrémentaux, undo, scoring. |
| `game.py` | Simulateur : deck physique, tours, hiver, multijoueur. |
| `search.py` | greedy, beam, MCTS à information imparfaite. |
| `bench.py` | Mesures reproductibles. |
| `tests/test_rules.py` | Oracle + cas unitaires + fuzzing. |
| `reference/value_policy.py` | `leaf_eval` pour MCTS : MLP absolu, contrastif pur, contrastif hybride (voir §4). |
| `reference/gen_pairwise_dataset.py`, `train_pairwise_model.py` | Génère/entraîne le modèle contrastif (`pairwise_model.joblib`). |
| `reference/card_strength.py`, `card_strength_mcts.py` | Force intrinsèque des cartes par retrait contrefactuel (greedy / MCTS). |
| `reference/strength_policy.py`, `bench_strength.py` | Tentative (négative) d'injecter la force des cartes dans une politique gloutonne. |
| `run_narrated_hybrid.py` | Rejoue une partie MCTS (config gagnante) coup par coup, avec la main et les alternatives explorées. |
| `run_stats_hybrid.py` | Statistiques agrégées (fréquence par carte) sur N parties MCTS. |

## 6. Utilisation

```bash
python -m pytest tests/ -q      # 14 tests, dont 1700 forêts aléatoires
python bench.py scoring
python bench.py policies
python bench.py mcts 400 40 8
python bench.py mcts_pairwise_hybrid 300 8 10   # config MCTS gagnante (voir §4)
```

```python
from engine import Forest, TREE_ID, DWELLER_ID, POS_ID, score_players

f = Forest()
i = f.add_tree(TREE_ID["BEECH"])
f.add_dweller(i, POS_ID["Right"], DWELLER_ID["ROE_DEER"])
print(f.score())

# En multijoueur, laisser score_players résoudre les majorités :
# en solo, LINDEN et le pic épeiche sont gratuitement maximaux.
print(score_players([f_joueur1, f_joueur2]))
```

## 7. Garde-fou

`tests/test_rules.py` compare le moteur optimisé à l'oracle sur **1 700 forêts
aléatoires** (dont 200 de taille fin de partie), en plus des cas unitaires
repris du dépôt de référence. C'est ce harnais qui rendra le portage `jnp`
sûr : chaque règle vectorisée devra passer les mêmes tests.
