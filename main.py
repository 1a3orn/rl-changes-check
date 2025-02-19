from vllm import LLM, SamplingParams


def load_llm(model_name):
    llm = LLM(
        model=model_name,
        dtype="float16",
    )
    return llm


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = load_llm("agentica-org/DeepScale-R-1.5B-Preview")