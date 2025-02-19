import json

from vllm import LLM, SamplingParams


def load_llm(model_path):
    llm = LLM(model=model_path, dtype="float16")
    return llm

def load_prompts(path):
    with open(path, "r") as f:
        data = json.load(f)

    converted = []
    for item in data:

        question_str = item["question"]
        answer_str = item["answer"]

        # Append some instructions about <think> and <answer>
        addition = (
            ""
            #"First, please think step-by-step about how to get the right answer. "
            #"You may use any technique you want to find the answer or check that it is right."
            #"Afterwards, write the answer and only the answer inside <answer>...</answer> tags."
        )
        full_prompt = f"{question_str}{addition}"
        converted.append({"prompt": full_prompt.trim(), "answer": answer_str})

    return converted



def main():

    print("Loading prompts...")
    prompts = load_prompts("./datasets/trash_math_train_questions.json")
    prompt_text = [item["prompt"] for item in prompts][:10]

    print("Loading model...")
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=3000)
    llm = load_llm("./agentica")

    CHUNK_SIZE = 2
    for i in range(len(prompt_text) // CHUNK_SIZE):
        print(f"Processing prompts {i} to {i + CHUNK_SIZE - 1} of {len(prompt_text) // CHUNK_SIZE}")
        outputs = llm.generate(prompt_text[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE], sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\n\nPrompt: {prompt!r}\nGenerated text: {generated_text!r}")


if __name__ == "__main__":
    main()
