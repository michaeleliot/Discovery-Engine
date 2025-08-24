class PromptSampler:
    def __init__(self, llm):
        self.llm = llm  # LLM instance used to generate new base prompt

    async def build(self, parent_program: str, base_prompt: str, inspiration: str) -> str:
        """
        Use the LLM to generate a new base prompt given the previous base prompt,
        the parent program, and an inspiration. Returns the combined prompt for diff generation.
        """
        if inspiration:
            new_base_prompt = await self.llm.generate_new_base_prompt(
                base_prompt=base_prompt,
                parent_program=parent_program,
                inspiration=inspiration
            )
        else:
            new_base_prompt = base_prompt

        # Combine into full prompt for diff generation
        combined_prompt = f"""
Base Prompt for this iteration:
{new_base_prompt}

Parent Program:
{parent_program}

Instructions:
Generate diffs to improve the parent program. Use the following format for all changes:

<<<<<<< SEARCH
# Original code block to be found and replaced
=======
# New code block to replace the original
>>>>>>> REPLACE

Make sure the final computed value to be evaluated is assigned to the variable `result`.
"""
        return combined_prompt, new_base_prompt
