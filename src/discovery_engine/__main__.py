import asyncio
import sys

from dotenv import load_dotenv

from .runnner import run

sys.set_int_max_str_digits(0)


# Load variables from .env
load_dotenv()

async def run_example():
  evaluator_program = """
def evaluator(program, result) -> dict:
  score = result if isinstance(result, int) else 0
  return {"score": score, "output": f"Custom evaluated {program}"}
"""

  initial_program = """
def compute():
  return 10
result = compute()
  """

  initial_base_prompt = """
  This prompt will be used to generate code, but it is unclear exactly what the evaluation
  metric is. You should write code that you think would maximize some kind of score. Keep the code relatively short and simple.
  Focus on producing code that is correct, well-structured, and likely to achieve a high score.
  """
  await run(evaluator_program, initial_program, initial_base_prompt)


if __name__ == "__main__":
    asyncio.run(run_example())