

from vllm import LLM
from vllm import SamplingParams
question_template = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n"
answer_template = "\n\n### 応答:\n"


def gen_prompt(q):
    return f"{question_template}{q}{answer_template}"


class TanukiBot:
    def __init__(self,
                 model_name="hatakeyama-llm-team/Tanuki-8B-Instruct",
                 max_model_len=5000,
                 gpu_memory_utilization=0.2
                 ):
        self.model_name = model_name

        print("initialing model...")
        self.llm = LLM(model=model_name, trust_remote_code=True,
                       max_model_len=max_model_len,
                       # tensor_parallel_size=2,
                       gpu_memory_utilization=gpu_memory_utilization,
                       )

    def ask(self, question):
        prompt = gen_prompt(question)
        prompts = [prompt]
        outputs = self.llm.generate(
            prompts,
            sampling_params=SamplingParams(
                temperature=0.5,
                max_tokens=1024,
            )
        )
        output = outputs[0]
        return output.outputs[0].text.strip()
