from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from discovery_engine.runnner import run

load_dotenv()
# Import your real dependencies here:
# from your_module import Database, LLM, PromptSampler, Evaluator, Explorer

app = FastAPI()

class InputData(BaseModel):
    evaluator_program: str
    initial_program: str
    initial_base_prompt: str

@app.post("/process")
async def process_strings(data: InputData):
    result = await run(
        data.evaluator_program,
        data.initial_program,
        data.initial_base_prompt,
    )
     # If `result` is a custom object, convert it
    if not isinstance(result, (dict, list, str, int, float, bool, type(None))):
        result = str(result)  # fallback: stringify

    return {"status": "completed", "result": result}
