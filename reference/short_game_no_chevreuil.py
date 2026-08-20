"""Analyse conditionnelle sur les forets sauvegardees par
short_game_breakdown_e.py (reference/short_game_e_rows.pkl) : question de
Mehdi (19/08) -- le Chevreuil domine largement l'ecart top/bottom sur une
partie courte, mais que faire si on n'en pioche pas (question de variance) ?

1. Distribution du score selon le nombre de Chevreuils obtenus (0, 1-2, 3+).
2. PARMI les forets a 0 Chevreuil (le "pire des cas" sur ce levier), quel
   autre levier differencie le mieux un bon score d'un mauvais -- le
   meilleur plan B mesurable.

Usage : python reference/short_game_no_chevreuil.py
"""
import pickle
import statistics
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
with open(HERE / "short_game_e_rows.pkl", "rb") as f:
    rows = pickle.load(f)  # liste de (score, breakdown_dict, n_cards)

print(f"{len(rows)} forets chargees\n")

# --- 1. score selon nb de Chevreuils (ROE_DEER points > 0 = au moins 1 pose) ---
# Le score ROE_DEER dans le breakdown est deja 3 x nb_Chevreuils_symbole x nb_cartes_symbole,
# donc on bucket sur les points eux-memes (proxy raisonnable de "combien de Chevreuils actifs").
buckets = defaultdict(list)
for sc, bd, n_cards in rows:
    rd = bd.get("ROE_DEER", 0)
    if rd == 0:
        key = "0 (aucun Chevreuil actif)"
    elif rd < 15:
        key = "1 (Chevreuil faible, <15 pts)"
    elif rd < 40:
        key = "2 (Chevreuil correct, 15-40 pts)"
    else:
        key = "3 (Chevreuil fort, 40+ pts)"
    buckets[key].append(sc)

print("=== Score total selon le niveau de Chevreuil obtenu ===")
for key in sorted(buckets):
    vals = buckets[key]
    print(f"  {key:35s} n={len(vals):4d}  score moyen {statistics.mean(vals):6.1f}  "
          f"mediane {statistics.median(vals):6.0f}  min {min(vals):4.0f}  max {max(vals):4.0f}")

# --- 2. parmi les forets SANS Chevreuil, qu'est-ce qui differencie top/bottom ? ---
zero_rd = [(sc, bd, n) for sc, bd, n in rows if bd.get("ROE_DEER", 0) == 0]
print(f"\n=== Parmi les {len(zero_rd)} forets SANS Chevreuil actif : quel est le meilleur plan B ? ===")
zero_rd.sort(key=lambda r: -r[0])
k = max(1, len(zero_rd) // 5)
top = zero_rd[:k]
bottom = zero_rd[-k:]
print(f"score moyen top20% (sans Chevreuil) : {statistics.mean(r[0] for r in top):.1f} "
      f"vs bottom20% : {statistics.mean(r[0] for r in bottom):.1f}")

all_sources = set()
for _, bd, _ in zero_rd:
    all_sources.update(bd.keys())
diffs = []
for src in all_sources:
    if src == "ROE_DEER":
        continue
    t = statistics.mean(bd.get(src, 0) for _, bd, _ in top)
    b = statistics.mean(bd.get(src, 0) for _, bd, _ in bottom)
    diffs.append((src, t, b, t - b))
diffs.sort(key=lambda x: -x[3])
print(f"{'source':28s} {'top20%':>8s} {'bottom20%':>10s} {'ecart':>8s}")
for src, t, b, d in diffs[:12]:
    print(f"{src:28s} {t:8.2f} {b:10.2f} {d:+8.2f}")

# score moyen general pour comparaison
print(f"\nscore moyen TOUTES forets (rappel) : {statistics.mean(r[0] for r in rows):.1f}")
print(f"score moyen forets SANS Chevreuil    : {statistics.mean(r[0] for r in zero_rd):.1f} "
      f"({len(zero_rd)}/{len(rows)} = {len(zero_rd)/len(rows)*100:.0f}% des forets)")
