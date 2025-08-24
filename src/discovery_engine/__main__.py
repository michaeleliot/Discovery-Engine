import asyncio
import sys
from typing import Any

from dotenv import load_dotenv

from .database import Database
from .enabler import Enabler
from .evaluator import Evaluator
from .llm import LLM
from .prompt_sampler import PromptSampler

sys.set_int_max_str_digits(0)


# Load variables from .env
load_dotenv()

def my_custom_eval(program: str, result: Any) -> dict:
    score = result if isinstance(result, int) else 0
    return {"score": score, "output": f"Custom evaluated {program}"}

async def run_example():
  # Seed database
  database = Database()
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
  llm = LLM()

  embedding = await llm.embed_program(initial_program)

  database.add_result(initial_program, my_custom_eval(initial_program, 10), initial_base_prompt, embedding)
  database.add_inspiration(parent_result_id=1, description="")

  # Initialize components

  prompt_sampler = PromptSampler(llm)
  evaluator = Evaluator(eval_fn=my_custom_eval)
  enabler = Enabler(database, prompt_sampler, llm, evaluator, iterations=2)

  # Run the evolution
  await enabler.run()


if __name__ == "__main__":
    asyncio.run(run_example())