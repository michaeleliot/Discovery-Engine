import asyncio
import time


class Explorer:
    def __init__(self, database, prompt_sampler, llm, evaluator, iterations=20):
        self.database = database
        self.prompt_sampler = prompt_sampler
        self.llm = llm
        self.evaluator = evaluator
        self.iterations = iterations

    async def run(self):
        for i in range(self.iterations):
            iter_start = time.perf_counter()
            print(f"\n=== Iteration {i+1}/{self.iterations} ===")

            parent_entry, inspirations, all_inspirations = self.database.sample()

            parent_id = parent_entry["id"]
            print(f"Chosen Parent Id {parent_id}")

            # Schedule all inspirations concurrently
            tasks = [
                asyncio.create_task(
                    self.explore_inspiration(parent_entry, inspiration, all_inspirations)
                )
                for inspiration in inspirations
            ]

            # Wait for all to finish before moving to the next parent
            await asyncio.gather(*tasks)

            iter_end = time.perf_counter()
            print(f"Iteration {i+1} took {iter_end - iter_start:.2f} seconds")


        return self.database.best()

    async def explore_inspiration(self, parent_entry: dict, inspiration: dict, all_inspirations: list[dict]):
        parent_program = parent_entry["program"]
        base_prompt = parent_entry["base_prompt"]

        # Capture the parent entry & score at the start of the iteration
        parent_score = parent_entry["results"].get("score", 0)
        # Build combined prompt (returns new base prompt as well)
        combined_prompt, new_base_prompt = await self.prompt_sampler.build(
            parent_program, base_prompt, inspiration
        )

        # Get LLM diff, apply it, evaluate the child
        diff, _ = await self.llm.generate(combined_prompt)
        child_program = self.llm.apply_diff(parent_program, diff)

        results = self.evaluator.execute(child_program)
        child_score = results.get("score", 0)
        if child_score < parent_score:
          # Regression: create inspirations for the parent
          parent_insps = await self.llm.generate_inspiration_regression(
              parent_base_prompt=base_prompt,
              parent_program=parent_program,
              parent_results=parent_entry["results"],
              child_program=child_program,
              child_results=results,
              existing_inspirations=all_inspirations
          )
        else:
            # No regression: still create inspirations for the parent
            parent_insps = await self.llm.generate_recommendation(
                base_prompt,
                parent_program,
                parent_entry["results"],
                existing_inspirations=all_inspirations
            )

        # Child always gets inspirations
        child_insps = await self.llm.generate_recommendation(
            new_base_prompt,
            child_program,
            results,
            existing_inspirations=all_inspirations
        )

        embedding = await self.llm.embed_program(child_program)

        result = self.database.add_result(child_program, results, new_base_prompt, embedding)

        self.database.mark_inspiration_as_used(inspiration=inspiration, result_id=result["id"])

        print(f"Inspirations Length Parent Insps:{len(parent_insps)}, Child Insps: {len(child_insps)}")

        for insp in parent_insps:
            self.database.add_inspiration(
                parent_result_id=parent_entry["id"],
                description=insp
            )
            print("\nParent Inspiration:\n", insp)
        for insp in child_insps:
            self.database.add_inspiration(
                parent_result_id=result["id"],
                description=insp
            )
            print("\nChild Inspiration:\n", insp)

