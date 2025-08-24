import numpy as np
from google import genai

from .project_types import Inspiration


class LLM:
    def __init__(self):
        self.client = genai.Client() # Gets API key from GEMINI_API_KEY

    async def embed_program(self, program: str):
        """Generate embedding vector for a program string."""
        response = await self.client.aio.models.embed_content(
            model="gemini-embedding-001",
            contents=program,
        )
        return np.array(response.embeddings[0].values)

    async def generate(self, base_prompt: str):
        response = await self.client.aio.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=base_prompt
        )
        diff_text = response.candidates[0].content.parts[0].text
        return diff_text, base_prompt

    def apply_diff(self, parent_program: str, diff: str):
        import re
        pattern = re.compile(
            r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
            re.DOTALL
        )

        updated_program = parent_program
        for match in pattern.finditer(diff):
            original_block = match.group(1).strip("\n")
            new_block = match.group(2).strip("\n")
            updated_program = updated_program.replace(original_block, new_block)
        return updated_program

    async def generate_inspiration_regression(
        self,
        parent_base_prompt: str,
        parent_program: str,
        parent_results: dict,
        child_program: str,
        child_results: dict,
        existing_inspirations=list[dict]
    ) -> list[str]:
        """
        Produce a short inspiration idea when a new child performed worse than the parent.
        The idea should hypothesize about the hidden evaluation criteria and suggest
        prompt updates that could improve score next time.
        """
        prompt = f"""
You are analyzing a code evolution experiment where the goal is to maximize an UNKNOWN (hidden) score.

CONTEXT (Parent performed BETTER than Child):
- Parent Base Prompt:
{parent_base_prompt}

- Parent Program:
{parent_program}

- Parent Results (incl. score):
{parent_results}

- Child Program (performed worse):
{child_program}

- Child Results (incl. score):
{child_results}

- Existing Updates/Inspirations:
{existing_inspirations}

TASK:
1) Diagnose why the Child likely underperformed relative to the Parent.
2) Hypothesize about the hidden scoring function (what it may reward/penalize).
3) Recommend 1–3 concrete, testable updates to the BASE PROMPT that better target the hidden criteria. Do not repeat any of the existing updates.
4) Summarize each recommendation as ONE concise "inspiration idea" that we can try next iteration. Do not repeat any of the existing inspirations.

OUTPUT FORMAT:
- Return ONLY a JSON list of inspirations.
- Each inspiration should be an object of the form: {{ "description": "<string>" }}
- The "description" should briefly capture the suspected scoring factors and how to update the base prompt.
"""
        response = await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[Inspiration],
            },
        )
        inspirations: list[Inspiration] = response.parsed
        return [inspiration.description for inspiration in inspirations]


    async def generate_new_base_prompt(self, base_prompt: str, parent_program: str, inspiration: str) -> str:
        """
        Call Gemini to create a new base prompt given previous prompt, program, and inspiration.
        """
        prompt = f"""
You are improving a code generation prompt.

Previous Base Prompt:
{base_prompt}

Parent Program:
{parent_program}

Inspiration Idea:
{inspiration["description"]}

Generate a new base prompt that incorporates the inspiration idea
and would lead to an improved program.
Return only the new prompt as plain text.
"""
        response = await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        new_prompt = response.candidates[0].content.parts[0].text
        return new_prompt

    async def generate_recommendation(self, base_prompt: str, program: str, results: dict, existing_inspirations: list[dict]) -> list[str]:
        """
        Ask Gemini to suggest future improvements with explicit reasoning
        about the UNKNOWN scoring function.
        """
        prompt = f"""
You are reviewing a code generation experiment where the evaluation score is HIDDEN/UNKNOWN.
Assume we want to maximize that hidden score and learn from this iteration.

Base Prompt:
{base_prompt}

Generated Program:
{program}

Evaluation Results (incl. score):
{results}

- Existing Updates:
{existing_inspirations}

TASK:
1) Infer what the hidden scoring function might reward or penalize based on the current outcome.
2) Propose 1–3 concise, testable updates to the BASE PROMPT that exploit those hypotheses. Do not repeat any of the existing updates.
3) Summarize each update as one short "inspiration idea" (1–3 sentences) to try next iteration.

OUTPUT FORMAT:
- Return ONLY a JSON list of inspirations.
- Each inspiration should be an object of the form: {{ "description": "<string>" }}
- The "description" should briefly capture the suspected scoring factors and how to update the base prompt.
"""

        response = await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[Inspiration],
            },
        )
        inspirations: list[Inspiration] = response.parsed
        return [inspiration.description for inspiration in inspirations]
