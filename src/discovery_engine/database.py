import random
from typing import Any

import numpy as np


class Database:
    def __init__(self, num_categories = 5, mutation_rate = 0.1, num_inspirations = 5, num_elites = 5):
        self.results = []  # main generated programs
        self.inspirations = []  # new ideas to try
        self._next_result_id = 1
        self._next_inspiration_id = 1
        self.num_categories = num_categories
        self.mutation_rate = mutation_rate
        self.num_inspirations = num_inspirations
        self.num_elites = num_elites

    def print_categories(self):
        """Print all categories and their associated results (ids + scores)."""
        print("\n=== Categories ===")

        # Group results by category
        categories = {}
        for entry in self.results:
            cat = entry.get("category", None)
            if cat is not None:
                categories.setdefault(cat, []).append(entry)

        # Print grouped results
        for cat, entries in categories.items():
            print(f"\nCategory {cat}:")
            for e in entries:
                score = e["results"].get("score", 0)
                print(f"  - Result {e['id']}: score={score}")

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def add_result(self, program, results, base_prompt, embedding):
        # Decide whether to create a new category
        if len(self.results) < self.num_categories:
            # make a new category
            category = f"cat-{len(self.results) + 1}"
            print("creating new category")
        else:
            # Assign category by embedding similarity
            sims = [
                (self.cosine_similarity(embedding, r["embedding"]), r)
                for r in self.results
            ]
            best_sim, closest_result = max(sims, key=lambda x: x[0])
            category = closest_result["category"]

        result_entry = {
            "id": self._next_result_id,
            "program": program,
            "results": results,
            "base_prompt": base_prompt,
            "embedding": embedding,
            "category": category,
        }

        self.results.append(result_entry)

        self._next_result_id += 1
        return result_entry

    def add_inspiration(self, parent_result_id, description, result_id=None):
        inspiration_entry = {
            "id": self._next_inspiration_id,
            "parent_result_id": parent_result_id,
            "description": description,
            "result_id": result_id
        }
        self.inspirations.append(inspiration_entry)
        self._next_inspiration_id += 1
        return inspiration_entry["id"]

    def sample(self):
      if not self.results:
          return None

      # pick a random category
      categories = list(set(r["category"] for r in self.results))
      category = random.choice(categories)

      # filter results from this category
      cat_results = [r for r in self.results if r["category"] == category]

      # pick top 5 in that category by score
      top_cat_results = sorted(cat_results, key=lambda r: r["results"].get("score", 0), reverse=True)[:self.num_elites]

      selected_entry = random.choice(top_cat_results)

      # Get all inspirations for this parent that don't yet have a generated result
      all_inspirations = [
          insp
          for insp in self.inspirations
          if insp["parent_result_id"] == selected_entry["id"]
      ]

      unused_inspirations = [
          insp for insp in all_inspirations if insp["result_id"] is None
      ][:self.num_inspirations]

      # === NEW: Cross-category inspiration swap (mutation) ===
      if random.random() < self.mutation_rate and len(categories) > 1:
          # pick a different category
          other_category = random.choice([c for c in categories if c != category])

          # filter results from that other category
          other_results = [r for r in self.results if r["category"] == other_category]

          # pick top 5 in the other category
          top_other_results = sorted(other_results, key=lambda r: r["results"].get("score", 0), reverse=True)[:self.num_elites]

          other_selected = random.choice(top_other_results)

          # swap inspirations (but keep the original selected_entry!)
          all_inspirations = [
              insp for insp in self.inspirations if insp["parent_result_id"] == other_selected["id"]
          ]
          unused_inspirations = [
              insp for insp in all_inspirations if insp["result_id"] is None
          ][:self.num_inspirations]

          print(f"[Mutation] Swapped inspirations from category {category} → {other_category}")

      # sample one of them at random
      return selected_entry, unused_inspirations, all_inspirations


    def best(self):
        if not self.results:
            return None
        best = max(self.results, key=lambda r: (r["results"].get("score", 0), r["id"]))
        return {
            "program": best["program"],
            "base_prompt": best["base_prompt"]
        }

    def mark_inspiration_as_used(self, inspiration: Any, result_id: int):
        """
        Updates the inspiration that matches `inspiration_description` and has no result_id yet,
        setting its `result_id` to the given result_id.
        """
        for insp_entry in self.inspirations:
            if insp_entry["id"] == inspiration["id"] and insp_entry["result_id"] is None:
                insp_entry["result_id"] = result_id
