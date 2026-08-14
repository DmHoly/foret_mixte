"""
Génère `cards.py` depuis le dépôt de référence soldag/forest-shuffle-scoring.

Pourquoi passer par une extraction plutôt que par une recopie : la version
précédente du projet avait 7 habitants avec des positions autorisées fausses,
introduites à la main. Une extraction mécanique ne peut pas dériver.

Prérequis : node + esbuild, et le dépôt cloné.

    git clone --depth 1 https://github.com/soldag/forest-shuffle-scoring ref
    npm install esbuild uuid lodash-es
    node_modules/.bin/esbuild tools/dump.ts --bundle --platform=node \
        --outfile=/tmp/dump.cjs --alias:@=./ref/src
    node /tmp/dump.cjs > tools/base_cards.json
    python tools/gen_cards.py

Le `dump.ts` correspondant importe les blueprints, filtre `gameBox === "BASE"`
et sérialise nom, coût, types et variantes (symbole, position, exemplaires).
Contrôle de cohérence attendu, aligné sur `deck.test.ts` : 66 arbres et 184
moitiés d'habitants.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TYPES = {
    "AMPHIBIAN": "Amphibian", "BAT": "Bat", "BIRD": "Bird",
    "BUTTERFLY": "Butterfly", "CLOVENHOOFED_ANIMAL": "ClovenhoofedAnimal",
    "DEER": "Deer", "INSECT": "Insect", "MUSHROOM": "Mushroom",
    "PAWED_ANIMAL": "PawedAnimal", "PLANT": "Plant", "TREE": "Tree",
}
POSITIONS = {"TOP": "Top", "BOTTOM": "Bottom", "LEFT": "Left", "RIGHT": "Right"}


def main():
    data = json.loads((HERE / "base_cards.json").read_text())
    trees = sorted((c for c in data["woodyPlants"] if c["isPartOfDeck"]),
                   key=lambda c: c["name"])
    dwellers = sorted(data["dwellers"], key=lambda c: c["name"])

    n_trees = sum(v["count"] for c in trees for v in c["variants"])
    n_halves = sum(v["count"] for c in dwellers for v in c["variants"])
    assert n_trees == 66, n_trees
    assert n_halves == 184, n_halves
    print(f"{len(trees)} arbres ({n_trees} exemplaires), "
          f"{len(dwellers)} habitants ({n_halves} moitiés)")


if __name__ == "__main__":
    main()
