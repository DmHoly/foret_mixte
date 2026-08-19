# Forêt Mixte : moteur, simulateur et bots

Implémentation Python du jeu de cartes *Forêt Mixte* (Lookout Games / Asmodee) :
moteur de scoring vérifié contre une référence externe, simulateur de partie
complet, et une lignée de bots qui se sont battus les uns contre les autres
jusqu'au meilleur actuel, **E**.

Les données de cartes sont **extraites mécaniquement** depuis
[soldag/forest-shuffle-scoring](https://github.com/soldag/forest-shuffle-scoring)
(`tools/gen_cards.py` → `cards.py`), pas recopiées à la main. Le moteur de
scoring (`engine.py`) est validé contre cette référence sur **1700 forêts
aléatoires** (`tests/test_rules.py`, 42 tests) — voir `scoring_ref.py` pour
l'oracle et le détail des corrections de règles apportées par rapport à une
implémentation naïve.

## Sommaire

- [Installation](#installation)
- [Évolution des bots](#évolution-des-bots)
- [Heuristiques adoptées, dans l'ordre](#heuristiques-adoptées-dans-lordre)
- [Le modèle Gradient Boosting (tiebreak)](#le-modèle-gradient-boosting-tiebreak)
- [Progression du score](#progression-du-score)
- [Tournoi Swiss entre les bots du dépôt](#tournoi-swiss-entre-les-bots-du-dépôt)
- [Guides stratégiques](#guides-stratégiques)
- [Jouer contre le bot](#jouer-contre-le-bot)
- [Structure du projet](#structure-du-projet)
- [Limitations connues](#limitations-connues)

## Installation

```bash
pip install -r requirements.txt  # numpy, scikit-learn, joblib (pour reference/)
python -m pytest tests/ -q       # 42 tests, dont 1700 forêts aléatoires
```

```python
from engine import Forest, TREE_ID, DWELLER_ID, POS_ID

f = Forest()
i = f.add_tree(TREE_ID["BEECH"])
f.add_dweller(i, POS_ID["Right"], DWELLER_ID["ROE_DEER"])
print(f.score())
```

## Évolution des bots

`search.py` et `reference/` racontent une lignée de bots, chacun construit
pour battre le précédent. Résumé chronologique — le détail complet (chaque
tentative, y compris les négatives) est dans `reference/MODELS.md`.

1. **Greedy nu** (delta de score exact, calculé par le moteur à chaque coup
   candidat) — sans correctif, ne plante jamais un arbre (aucun gain
   immédiat) et plafonne très bas. Une prime de pose d'arbre est nécessaire
   dès le départ pour que le greedy fonctionne du tout.
2. **Beam** (recherche en faisceau sur la main, adversaire et pioche
   ignorés) — perd contre le greedy tout simple. Regarder plus loin sans
   tenir compte de l'incertitude sur-optimise des lignes qui ne se
   réalisent jamais.
3. **MCTS** (arbre de recherche à information imparfaite, ISMCTS) — bat le
   greedy grâce à la recherche arborescente, à condition d'un budget de
   simulation suffisant (plusieurs centaines d'itérations). Rollouts
   guidés par le greedy + un modèle de valeur contrastif appris
   (`value_policy.make_pairwise_hybrid_leaf_eval`) pour rester rapide.
4. **L'ère des heuristiques qui battent MCTS (B)** — deux heuristiques
   ajoutées au greedy (ciblage des cartes fortes en Clairière, urgence de
   pioche) le font **battre MCTS en tête-à-tête direct**, de façon
   répétée sur plusieurs tournois. Contre-intuitif mais mesuré : sur ce
   jeu, un greedy bien réglé résiste étonnamment bien à l'ajout d'une
   recherche arborescente. B devient la référence du dépôt.
5. **Prime de pose d'arbre pilotée par le déficit de slots** — remplace
   une décroissance générique (`1/(1+n_trees)`) par une prime qui suit le
   vrai besoin de placement (habitants en main vs. slots déjà libres).
   Diagnostic en cours de route : planter un arbre rapporte en fait des
   points concrets dans 99.6% des cas réels (+7 pts en moyenne) — la
   prémisse "un arbre ne rapporte presque rien" n'était vraie qu'à
   l'ouverture.
6. **Le comparateur Gradient Boosting (E)** — un modèle non linéaire
   entraîné à départager les candidats presque à égalité de gain exact,
   branché comme second tour de décision dans `greedy_action`. **Bat B
   nettement (74-76/100, ~4.7 écarts-types)**, devient le meilleur bot du
   dépôt à ce jour — voir la section dédiée ci-dessous.

Chaque tentative de recherche "plus pure" pour dépasser B a échoué :
MCTS sans heuristique (8/30), modèle de valeur absolu (2/30), modèle
pairwise mal formulé (1/30), bootstrap MCTS façon AlphaZero (7/30). Le
motif qui se dégage sur six tentatives consécutives : améliorer un
modèle de valeur en optimisant sa précision hors ligne ne se traduit
quasiment jamais par une meilleure décision réelle — voir
`reference/MODELS.md` pour le détail de chacune.

## Heuristiques adoptées, dans l'ordre

| # | Heuristique | Remplace / ajoute | Effet mesuré |
|---|---|---|---|
| 1 | Clairière forte (`choose_draw_source`) | "toujours la carte la moins chère" | cible en priorité une carte forte connue si disponible — +18 pts en moyenne contre +3 pour une pioche aveugle équivalente |
| 2 | Urgence Clairière (`CLEARING_URGENCY_BONUS`) | rien (nouveau) | pioche plutôt que jouer un coup médiocre quand une carte forte est contestée en Clairière |
| 3 | Prime de pose d'arbre par déficit de slots | `tree_bonus / (1 + n_trees)` | neutre en remplissage global, gain concentré sur les 3 premiers arbres ; +6.2 pts/2000 parties contre B, +10.6 pts/400 contre E |
| 4 | Tiebreak Gradient Boosting | rien (nouveau, opt-in) | départage les candidats à moins de 3 pts du meilleur delta exact ; **E bat B 74-76/100** |

## Le modèle Gradient Boosting (tiebreak)

**Pourquoi pas un modèle linéaire.** Le modèle de valeur contrastif linéaire
(`pairwise_model.joblib`) est correctement formulé (régresse une différence
de gain entre deux candidats, pas un score absolu) mais plafonne : sur les
paires **serrées** (delta exact < 3 pts) — celles où la décision est la
plus difficile — sa précision de signe tombe **sous le hasard (47.7%)**.
Diagnostiqué comme un plafond de modèle, pas de features : un Gradient
Boosting entraîné sur les mêmes 78 features gagne **~10 points de
précision** sur cette tranche précise (57.3-57.7%, robuste sur 6
configurations d'hyperparamètres testées en validation croisée).

**Pourquoi un comparateur, pas un `leaf_eval`.** Un modèle d'ensemble
d'arbres prédit correctement une DIFFÉRENCE entre deux candidats, mais ne
se décompose pas en fonction de valeur par état (`arbre(a) - arbre(b) !=
arbre(a-b)`) — le brancher comme `leaf_eval` MCTS (qui a besoin d'une
valeur par état) reproduirait le piège qui a fait échouer un MLP siamois
plus tôt. Intégré à la place directement dans `greedy_action(...,
tiebreak=...)` : le delta exact reste le classement principal, seuls les
candidats à moins de 3 pts du meilleur sont départagés par le modèle, tous
comparés en un seul appel batché ("comparaison en étoile").

**Coût.** Regrouper les comparaisons d'un même tour en un seul appel
`model.predict()` (l'overhead est quasi fixe par appel, pas proportionnel
au nombre de lignes) fait passer le surcoût de ×140 à ×7.6 par rapport au
greedy nu — 0.34s/partie complète avec tiebreak actif d'un côté. Encore
trop cher pour un rollout MCTS (appelé des dizaines de fois par
simulation), mais largement viable pour des décisions réelles.

**Résultat.** E (greedy + ce tiebreak) bat B 74-76/100 selon la version,
écart moyen +50 à +66 pts (~4.7 écarts-types) — le premier résultat de
toute l'investigation à effectivement battre B. Branché aussi dans
`MCTS.choose()` (une fois, en fin de décision, pas dans le rollout) : bat
l'ancien MCTS, mais pas mieux que E — pas de nouveau meilleur bot par ce
chemin.

## Progression du score

Score moyen en auto-jeu (le bot affronte une copie de lui-même), mesuré
sur des parties récentes :

| Bot | Score moyen (self-play) |
|---|---|
| Historique, sans mécanique Clairière | 212 |
| BEAM | 409.9 |
| **D** — MCTS 150it, sans tiebreak | 298.1 |
| **F** — MCTS 150it + tiebreak GBM | 333.5 |
| **B** — greedy + heuristiques | 456.7 |
| **E** — greedy + heuristiques + tiebreak GBM | 450.7 |

Note importante : le score absolu **ne classe pas les bots entre eux**.
MCTS optimise une récompense de **rang normalisé** entre joueurs, pas le
score brut (backpropager le score brut ferait apprendre la variance du
tirage plutôt que la qualité des coups) — cohérent avec des scores
self-play plus bas pour D/F malgré un jeu recherché. Le vrai signal de
force est le **taux de victoire en tête-à-tête direct** (voir les sections
précédentes et `reference/MODELS.md`), pas la moyenne self-play.

## Tournoi Swiss entre les bots du dépôt

`reference/gen_swiss_tournament.py` fait s'affronter les 5 bots du dépôt
en tournoi Swiss (appariement par classement courant, sans répétition
d'adversaire quand possible, bye tournant si nombre impair) :

```bash
python reference/gen_swiss_tournament.py 4 4   # 4 rondes, 4 parties/match
```

Classement d'un run (4 rondes, 4 parties/match, sièges alternés) :

| # | Bot | Points | Marge cumulée |
|---|---|---|---|
| 1 | **E** | 4.0 | +1813 |
| 2 | B | 3.0 | +283 |
| 3 | BEAM | 2.0 | -107 |
| 4 | F | 2.0 | -311 |
| 5 | D | 1.0 | -1678 |

E en tête et D en dernier sont cohérents avec le reste de cette page. La
position de F (sous BEAM) est probablement du bruit d'échantillon (4
parties/match seulement, MCTS coûteux à faire tourner en volume) plutôt
qu'un vrai signal — pour la mesure rigoureuse F vs E, voir
`reference/MODELS.md` ("Tiebreak branché dans MCTS"). Ce tournoi est un
aperçu ludique, pas la preuve statistique : les tableaux de gating
(n=100-2000, écarts-types calculés) dans les sections précédentes et
`reference/MODELS.md` restent la référence.

## Guides stratégiques

Cinq pages HTML autonomes (aucun serveur requis), générées depuis des
parties simulées et pensées pour un joueur humain :

| Guide | Contenu |
|---|---|
| [docs/combo_guide.html](docs/combo_guide.html) | Tous les combos classés par espérance de points (probabilité × gain) |
| [docs/tactical_guide.html](docs/tactical_guide.html) | Ce que MCTS privilégie par rapport au greedy, et pourquoi |
| [docs/technique_guide.html](docs/technique_guide.html) | Chaque carte suivie coup par coup : tempo d'ouverture, immédiateté de pose, taux de défausse |
| [docs/decisive_guide.html](docs/decisive_guide.html) | Sur les meilleures parties de E, distingue les choix évidents des choix vraiment décisifs (là où le tiebreak change la décision) |
| [docs/strategic_guide.html](docs/strategic_guide.html) | Synthèse : gros leviers de score, Fouine, Lynx vs Sanglier, valeurs contrefactuelles vraies |

## Jouer contre le bot

```bash
python play_vs_bot.py            # contre E (le plus fort), tu commences
python play_vs_bot.py B 1        # contre le greedy simple, tu joues en second
python play_vs_bot.py E 0 12345  # graine fixe, pour rejouer la même partie
python play_vs_bot.py F          # contre MCTS + tiebreak, le plus lent à jouer
```

Chaque partie est journalisée dans `human_vs_bot_log.jsonl` pour suivre ta
progression sur plusieurs sessions.

## Structure du projet

| Fichier | Rôle |
|---|---|
| `cards.py` | Données des cartes. **Généré**, ne pas éditer à la main. |
| `scoring_ref.py` | Oracle : transcription littérale du moteur de référence. |
| `engine.py` | Moteur de scoring incrémental (`Forest.score()`, undo). |
| `game.py` | Simulateur : deck, tours, hiver, paiement, effets de carte. |
| `search.py` | Politiques : greedy, beam, MCTS. |
| `bench.py` | Benchmarks reproductibles (scoring, politiques, MCTS). |
| `play_vs_bot.py` | Partie interactive contre B/D/E/F. |
| `tests/test_rules.py` | Oracle + cas unitaires + fuzzing (1700 forêts). |
| `reference/` | Tout l'outillage d'expérimentation (modèles, diagnostics, générateurs de guides) — voir `reference/MODELS.md` pour l'index complet et l'historique de chaque tentative, positive ou négative. |
| `docs/` | Les 5 guides stratégiques ci-dessus. **Générés**, ne pas éditer à la main. |

## Limitations connues

- **Paiement et choix de la carte de Clairière** restent des heuristiques
  (`choose_payment`, `choose_draw_source`), pas des décisions de l'arbre
  de recherche — le plus gros trou de modélisation restant pour la
  qualité de jeu.
- **Choix de la moitié sacrifiée** en cas de carte à deux moitiés : prise
  actuellement comme "la première qui correspond", pas une vraie décision.
- **3+ joueurs** : le moteur supporte N joueurs, mais MCTS n'a été validé
  qu'à 2 (récompense par rang non testée en configuration non-alternée).
- **Parallélisation racine** (multiprocessing) : non implémentée.
