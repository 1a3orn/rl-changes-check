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
        full_prompt = f"{question_str} {addition}"
        converted.append({"prompt": full_prompt.strip(), "answer": answer_str})

    return converted

# Extracts the answer from the boxed{11054} format
def boxed_extractor(text):
    text = text.lower()
    try:
        return text.split("boxed{")[1].split("}")[0]
    except:
        try:
            return text.split("boxed {")[1].split("}")[0]
        except:
            return None

def get_is_correct(answer, correct_answer):
    if answer is None:
        return False
    answer = answer.lower()
    correct_answer = correct_answer.lower()
    try:
        return eval(answer) == eval(correct_answer)
    except:
        return answer == correct_answer

models = [
    ("agentica", boxed_extractor),
    ("deepseek", boxed_extractor),
    ("qwen", boxed_extractor),
]


def main():

    print("Loading prompts...")
    prompts = load_prompts("./datasets/trash_math_train_questions.json")
    prompt_text = [item["prompt"] for item in prompts][:25]

    for model_path, extractor in models:

        print(f"Loading model {model_path}...")
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=-1, max_tokens=6000)
        llm = load_llm(model_path)

        count_correct = 0
        count_total = 0

        outputs = llm.generate(prompt_text, sampling_params)

        record = []
        for j, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            answer_given = extractor(generated_text)
            answer_correct = prompts[j]["answer"]
            is_correct = get_is_correct(answer_given, answer_correct)
            count_correct += int(is_correct)
            count_total += 1
            record.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "answer_given": answer_given,
                "answer_correct": answer_correct,
                "is_correct": is_correct
            })
            #print("Answer given: ", answer_given)
            #print("Answer correct: ", answer_correct)
            #print("Is correct: ", is_correct)
            #print("")

        with open(f"results_{model_path}.json", "w") as f:
            json.dump(record, f)
        print("\n\n")
        print(f"Model: {model_path}")
        print(f"Accuracy: {count_correct / count_total}")

        # clear up memory
        del llm
        del outputs
        del record



if __name__ == "__main__":
    main()
