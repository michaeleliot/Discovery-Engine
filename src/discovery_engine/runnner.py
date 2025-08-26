from .database import Database
from .evaluator import Evaluator
from .explorer import Explorer
from .llm import LLM
from .prompt_sampler import PromptSampler


async def run(evaluator_program, initial_program, initial_base_prompt):
  database = Database()
  llm = LLM()

  embedding = await llm.embed_program(initial_program)

  prompt_sampler = PromptSampler(llm)
  evaluator = Evaluator(eval_fn_text=evaluator_program)
  explorer = Explorer(database, prompt_sampler, llm, evaluator, iterations=2)
  database.add_result(initial_program, evaluator.execute(initial_program), initial_base_prompt, embedding)
  database.add_inspiration(parent_result_id=1, description="")

  return await explorer.run()