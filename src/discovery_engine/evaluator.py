class Evaluator:
    def __init__(self, eval_fn_text: str):
        eval_fn = self.getEvaluatorFunctionFromText(eval_fn_text)
        self.eval_fn = eval_fn

    def getEvaluatorFunctionFromText(self, func_code):
        namespace = {}
        exec(func_code, namespace)
        f = namespace["evaluator"]
        return f

    def execute(self, program: str) -> dict:
        local_env = {}
        try:
            exec(program, {}, local_env)
            result_value = local_env.get("result", None)
        except Exception as e:
            return {"score": -1, "error": str(e)}

        return self.eval_fn(program, result_value)
