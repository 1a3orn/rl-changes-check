from vllm import LLM, SamplingParams


def load_llm(model_path):
    llm = LLM(model=model_path, dtype="float16")
    return llm



def main():
    prompts = [
        "Blork",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = load_llm("./agentica")

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
