from typing import Any, Callable


class Evaluator:
    def __init__(self, eval_fn: Callable[[str, Any], dict]):
        self.eval_fn = eval_fn

    def execute(self, program: str) -> dict:
        local_env = {}
        try:
            exec(program, {}, local_env)
            result_value = local_env.get("result", None)
        except Exception as e:
            return {"score": -1, "error": str(e)}

        return self.eval_fn(program, result_value)
