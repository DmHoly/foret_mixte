# Revue de projet : Forêt Mixte, moteur Python et portage MCTS/JAX

Périmètre audité : les 5 fichiers de l'archive (`engine.py`, `cards_data.py`, `deck_data.py`, `README.md`, `GAME_DATA.md`), soit 595 lignes.

---

## 1. Verdict

**Le portage JAX est faisable techniquement, mais il est prématuré et il vise le mauvais goulot.**

Trois conclusions séparées :

| Question | Réponse |
|---|---|
| Peut-on vectoriser `score_forest` en JAX ? | Oui, sans difficulté réelle. 2 à 3 jours. |
| Le moteur actuel est-il un oracle de référence fiable ? | **Non.** Au moins 4 cartes scorent 0 à tort, et un plafond de seuil est faux. |
| `mctx` répond-il au besoin décrit ? | **Non tel que décrit.** Le plan repose sur une hypothèse fausse sur `mctx`. |

Le point 3 est le vrai sujet de cette revue. Les points 2 et 3 rendent l'étape 2 du plan (« porter carte par carte en testant contre `engine.py` comme oracle ») dangereuse : on figerait des bugs dans une implémentation 10x plus coûteuse à corriger.

---

## 2. Ce qui tient

- **La structure de données est propre.** Séparation cartes / deck / moteur, `DWELLER_TYPESETS` précalculé, `copy()` manuel au lieu de `deepcopy`. Ce sont de bons réflexes.
- **Cohérence des données vérifiée.** J'ai audité : 0 dweller sans entrée de copies, 0 entrée orpheline, 0 arbre inexistant cité dans un `slots`, arithmétique du deck exacte (66 + 184 = 250). Les deux sources de vérité sur les copies d'arbres (`cards_data.TREES[...]["copies"]` et `deck_data.TREE_COPIES`) coïncident aujourd'hui.
- **Le moteur tourne** et produit des scores plausibles en structure.
- **La démarche est la bonne** : reconstruire depuis un moteur open-source vérifié plutôt que depuis les règles en prose.

---

## 3. Défauts factuels trouvés dans le moteur

### 3.1 Huit dwellers n'ont aucune règle de score (bloquant)

Analyse statique de `engine.py` croisée avec `DWELLERS_RAW` :

| Carte | Coût | Exemplaires | Type | Statut |
|---|---|---|---|---|
| BROWN_BEAR | 3 | 3 | PawedAnimal | **anomalie** |
| MOLE | 2 | 2 | PawedAnimal | **anomalie** |
| RACCOON | 1 | 4 | PawedAnimal | **anomalie** |
| VIOLET_CARPENTER_BEE | 1 | 4 | Insect | **anomalie** |
| CHANTERELLE, FLY_AGARIC, PARASOL_MUSHROOM, PENNY_BUN | 2 | 2 chacun | Mushroom | légitime (moteurs de pioche) |

Les champignons à 0 point sont documentés dans `GAME_DATA.md` et corrects. Les 4 autres ne le sont pas : `BROWN_BEAR` coûte 3, le coût le plus élevé du jeu, et il n'y a que 7 cartes à coût 3 dans tout le deck. Une carte à 0 point à ce prix n'existe pas dans un jeu publié. Cela représente **13 exemplaires du deck silencieusement neutralisés**, soit 5,2 % des cartes jouables.

Conséquence directe : toute politique entraînée ou évaluée sur ce moteur apprend à ne jamais jouer ces cartes. Le biais est systématique, pas du bruit.

### 3.2 Plafond de seuil incohérent avec le deck

```python
pts = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49}   # Horse Chestnut
```

La table s'arrête à 7 alors que `deck_data.py` déclare **11 exemplaires** de Horse Chestnut, et c'est la carte la plus fréquente du jeu. Soit le plafond à 7 est la vraie règle (et il faut le documenter comme tel), soit la loi est n² et 8 à 11 châtaigniers sont sous-scorés de 15 à 72 points. À vérifier avant portage : c'est exactement le genre de détail qui coûte cher à débusquer une fois en `jnp`.

### 3.3 Table papillons partiellement morte

```python
fly_pts = {2: 3, 3: 6, 4: 12, 5: 20, 6: 35, 7: 55, 8: 80}
```

Il n'y a que 5 espèces de papillons dans le jeu de base. Les clés 6, 7, 8 sont inatteignables. Plus important : `fly_pts.get(1, 0)` renvoie 0, donc **un papillon isolé vaut 0 point**. C'est peut-être la règle, mais ce n'est écrit nulle part et c'est une décision de modélisation lourde de conséquences pour une politique.

### 3.4 Divergence solo / multijoueur non quantifiée

`engine.py` assume `is_linden_majority=True` et `is_tree_majority=True` par défaut. Le docstring le signale honnêtement, mais l'effet est cumulé avec deux autres biais :

- `add_dweller` incrémente `self.cave` systématiquement, donc +1 point par dweller joué, automatique et non optionnel ;
- pas de concurrence pour les cartes du deck partagé.

Résultat mesuré : sur une forêt de fin de partie générée aléatoirement (25 arbres, 70 dwellers), `score_forest` renvoie **973 points**. `GAME_DATA.md` cite 250 à 320 points pour un bon joueur humain à 2 joueurs, et vos propres simulations 400 à 500. Le document conclut que c'est « cohérent ». Ça ne l'est pas : un facteur 1,5 à 3 sur la métrique cible signifie que **le baseline contre lequel vous mesurerez le gain du MCTS/JAX n'est pas comparable à une partie réelle**. Optimiser contre cette fonction objectif optimise pour un jeu qui n'est pas Forêt Mixte.

### 3.5 Incohérences de documentation

- README §1 : optimisation « ~4x plus rapide ». Docstring de `engine.py` : « Gain mesuré : ~10x ». Les deux ne peuvent pas être vraies.
- « 58 cartes » compte `SAPLING`, qui n'est pas dans le deck. Il y a 8 arbres jouables + 49 dwellers = 57 entités jouables.
- `engine.py` référence `multiplayer.py` dans deux commentaires. Le fichier n'est pas dans l'archive.

### 3.6 Dette de packaging

- `cards_data.py` et `deck_data.py` font un `print()` **à l'import**. Inacceptable dans une boucle `jit`/test/benchmark.
- Imports plats (`from cards_data import ...`), pas de package, pas de `requirements.txt`, pas de `__init__.py`.
- Aucun test. Pour un projet dont l'étape suivante est « tester chaque règle contre `engine.py` comme oracle », c'est le manque le plus coûteux.
- Duplication des copies d'arbres entre `cards_data` et `deck_data` : deux sources de vérité qui vont dériver.
- Dans `score_forest`, `fully_occupied_trees = 0` est une affectation morte (recalculée 15 lignes plus bas).

### 3.7 Fichiers manquants pour un handoff

L'archive présente le projet comme un point de reprise, mais il manque tout ce qui justifie la décision : `multiplayer.py`, les trois politiques (naïve, lookahead, MCTS-léger), le simulateur de parties, et les mesures citées (score moyen, variance, 15 s/partie). **La motivation du portage n'est pas reproductible depuis cette archive.**

---

## 4. Le trou dans le plan JAX

C'est le point principal.

Le README écrit : *« un vrai MCTS avec arbre persistant + UCB, et surtout des milliers de simulations en parallèle sur GPU/TPU via JAX »* et propose `mctx`.

**`mctx` ne fait pas de rollouts.** C'est une implémentation d'AlphaZero / MuZero / Gumbel MuZero. La valeur d'un nœud ne vient pas d'une simulation aléatoire jusqu'à la fin de partie, elle vient du champ `value` que **vous** fournissez dans `RootFnOutput` et `RecurrentFnOutput`. Il n'y a aucune étape de simulation dans la bibliothèque.

Conséquence : le MCTS-léger actuel (un niveau + rollouts aléatoires) ne se « porte » pas sur `mctx`. Il faut choisir une des deux voies :

**Voie A, fonction de valeur écrite à la main.** `value = score_forest(état) + potentiel heuristique`. Pas d'entraînement, pas de GPU obligatoire, `mctx` utilisable en quelques jours une fois le moteur vectorisé. Réaliste. C'est ce que je recommande.

**Voie B, réseau de valeur entraîné par self-play.** C'est un projet d'apprentissage par renforcement complet : architecture, boucle de self-play, replay buffer, checkpointing, tuning. Compte en semaines à mois, pas en jours, et il faut un GPU. Le gain sur un jeu de cette taille n'est pas garanti.

Le README décrit le coût de la voie A et l'ambition de la voie B.

### 4.1 Trois autres obstacles non mentionnés dans le README

**a) `mctx` ne gère pas 3 à 5 joueurs.** Le backup de valeur est mono-agent (`discount=+1`) ou à somme nulle à deux joueurs alternés (`discount=-1`). Pour N > 2 il faut du max^n ou du paranoid, non supportés nativement. Or l'objectif §2 dit explicitement « multijoueur ». À 2 joueurs c'est jouable ; à 3+ il faut soit traiter le jeu comme mono-agent (défendable, l'interaction dans Forêt Mixte est limitée), soit écrire son propre backup. À trancher **avant** de choisir la représentation d'état, pas après.

**b) L'aléa de la pioche impose les nœuds de chance.** Le deck est stochastique. `mctx.stochastic_muzero_policy` existe et gère les afterstates, mais il faut un `num_chance_outcomes` fixe (58 types de cartes, ce qui double l'arbre) et l'implémentation est connue pour être lente sur ce point. L'alternative est la déterminisation (fixer un ordre de deck par simulation), plus simple et souvent suffisante.

**c) Le choix du paiement est une décision, pas un détail.** Jouer une carte à coût 2 implique de choisir *quelles* 2 cartes défausser, et le bonus couleur dépend de ce choix. Ce n'est nulle part dans le plan §3. Soit vous l'intégrez à l'espace d'actions (explosion combinatoire), soit vous le figez par heuristique (et vous perdez la mécanique de bonus couleur, déjà partiellement non implémentée d'après §4).

**d) L'espace d'actions n'est pas dimensionné.** `mctx` exige un `num_actions` fixe. Ici : (carte en main × arbre × position) + piocher, soit environ 10 × 25 × 4 + 1 ≈ 1000 actions. C'est gérable mais ça pilote la mémoire de l'arbre (`num_simulations × num_actions` par nœud). Ce dimensionnement devrait être l'étape 1 du plan, pas un sous-produit de l'étape 1 actuelle.

### 4.2 Ce qui est plus facile que le README ne le dit

Le README qualifie `score_forest` vectorisé de « morceau le plus long ». En volume, oui. En difficulté, non : les ~50 règles se ramènent à 6 primitives seulement (comptage par nom, comptage par type, seuil, condition arbre-porteur, adjacence de slot, comptage par position). Tout est `gather` + masque.

Le seul point qui ressemble à une vraie difficulté, la boucle `while` des papillons, a une **forme close** :

```
score = Σ_{r=0}^{max_count-1}  fly_pts[ #{espèces i : count_i > r} ]
```

Avec 5 espèces et 4 exemplaires maximum, c'est une matrice statique 4 × 5. Entièrement vectorisable, borne fixe, aucune boucle dynamique. Idem pour Horse Chestnut (n² ou `searchsorted`) et pour les tables à seuils.

---

## 5. Le goulot n'est pas là où vous le cherchez

Mesures faites sur cette machine (mono-cœur, forêt de fin de partie : 25 arbres, 70 dwellers) :

| Mesure | Valeur |
|---|---|
| `score_forest` complet | 178 µs |
| dont `ct()` (comptage par type) | ~86 µs, soit **48 %** |
| `ct()` par intersection de sets (actuel) | 7,2 µs/appel |
| `ct()` par masque de bits | 2,3 µs/appel (**x3,2**) |
| Appels `ct()` par scoring | ~12 |

`ct()` reconstruit `set(types_tuple)` **à chaque appel**, à l'intérieur de la boucle sur les dwellers, et recalcule le même comptage pour chaque instance d'une même carte. Deux corrections triviales, en Python pur, sans dépendance :

1. Masques de bits entiers au lieu de `frozenset` : x3,2 mesuré sur `ct()`.
2. Mémoïsation de `ct()` par tuple de types dans le scope de `score_forest` : les 12 appels tombent à 6 distincts au plus.

Combinées, **2 à 3x supplémentaires sans JAX**, en une heure de travail. Un scoring incrémental (ne recalculer que le delta induit par la carte posée) donnerait un ordre de grandeur de plus, et c'est le bon design pour des rollouts de toute façon.

Autrement dit : avant d'invoquer le GPU, il reste un facteur 10 sur la table en Python pur. Ça ne disqualifie pas JAX (la parallélisation batch reste le vrai levier), mais ça change le calcul de retour sur investissement de l'étape 2.

---

## 6. Chiffrage

| Chantier | Effort | Risque |
|---|---|---|
| Corriger les 4 cartes sans règle + vérifier les seuils | 1 à 2 j | faible, mais **prérequis absolu** |
| Suite de tests de non-régression sur le scoring | 1 j | faible |
| Optimisation Python pure (bitmask, mémoïsation, delta) | 1 à 2 j | faible |
| Nettoyage packaging (imports, prints, source unique) | 0,5 j | faible |
| `score_forest` vectorisé en `jnp` + tests contre l'oracle | 2 à 3 j | faible |
| Représentation d'état à taille fixe + transition | 3 à 5 j | moyen (partage de slots, paiement) |
| Espace d'actions + masquage de légalité | 2 à 3 j | moyen |
| Intégration `mctx`, voie A (valeur heuristique) | 3 à 5 j | moyen |
| Intégration `mctx`, voie B (réseau entraîné) | semaines à mois | élevé |
| Support 3+ joueurs propre | non chiffrable sans arbitrage | élevé |

**Voie A complète, à partir d'un moteur corrigé : 3 à 4 semaines à temps partiel.** C'est raisonnable. Voie B : autre projet.

---

## 7. Ordre recommandé

L'ordre du README (§6) commence par la représentation d'état. C'est l'inverse de ce qu'il faut faire : cette représentation dépend de décisions non prises (nombre de joueurs, gestion du paiement, taille de l'espace d'actions) et elle est bâtie sur un oracle faux.

1. **Trancher les deux questions de cadrage** : combien de joueurs, et le paiement est-il une action ou une heuristique ? Tout le reste en dépend.
2. **Réparer l'oracle.** Les 4 cartes sans règle, le plafond Horse Chestnut, le papillon isolé. Ré-étalonner ensuite le score simulé contre les 250-320 points humains ; si l'écart persiste, le modèle solo est faux et il faut le savoir maintenant.
3. **Écrire les tests** (forêts figées, scores attendus à la main pour une dizaine de configurations couvrant chaque primitive). Sans eux, l'étape de portage n'a pas d'oracle exploitable.
4. **Optimiser en Python pur** et re-mesurer les trois politiques. Il est possible que le coût de calcul cesse d'être le problème, ce qui reposerait entièrement la question du portage.
5. **Réunir les fichiers manquants** (`multiplayer.py`, les politiques, les benchmarks) dans le dépôt.
6. Seulement ensuite : représentation d'état, `score_forest` en `jnp`, transition, `mctx` voie A.

---

## 8. La question à se poser

Le README justifie le portage par le coût de calcul : 15 s par partie. Mais le vrai objectif est « un bot capable de jouer Forêt Mixte en multijoueur ». Ce sont deux objectifs différents.

Si l'objectif est **un bot qui joue bien**, la contrainte dominante est la fidélité des règles, pas la vitesse. Les défauts de la section 3 pèsent plus lourd qu'un facteur 100 sur le nombre de simulations : un MCTS profond sur un modèle faux converge plus vite vers une mauvaise stratégie.

Si l'objectif est **d'apprendre JAX et `mctx` sur un support motivant**, le projet est un excellent terrain et la faisabilité est bonne. Mais alors il faut l'assumer explicitement et ne pas mesurer le succès en points de score.

Ces deux objectifs mènent à des ordres de priorité opposés. Le README n'a pas tranché.
