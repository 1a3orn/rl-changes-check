import json
from tabulate import tabulate

from vllm import LLM, SamplingParams


def load_llm(model_path):
    llm = LLM(model=model_path, dtype="float16")
    return llm

def load_prompts(path, with_boxed_instructions=False):
    with open(path, "r") as f:
        data = json.load(f)

    converted = []
    for item in data:

        question_str = item["question"]
        answer_str = item["answer"]

        # Append some instructions about <think> and <answer>
        addition = (
            "Put the answer in the format of boxed{...} with the answer inside the brackets." if with_boxed_instructions else ""
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

def main_dataset(dataset_path):
    all_results = []  # Store results from all runs

    for model_path, extractor in models:
        print(f"Loading model {model_path}...")
        llm = load_llm(model_path)

        for with_boxed_instructions, temperature in [
            (True, 0.4),
            (False, 0.4),
            (True, 0.8),
            (False, 0.8),
        ]:
            prompts = load_prompts(dataset_path, with_boxed_instructions=with_boxed_instructions)
            prompt_text = [item["prompt"] for item in prompts][:10]

            sampling_params = SamplingParams(temperature=temperature, top_p=0.95, top_k=-1, max_tokens=6000)
            
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

            accuracy = count_correct / count_total
            
            # Save individual run results
            with open(f"results_{model_path}.json", "w") as f:
                json.dump(record, f, indent=2)
                
            # Add summary to aggregated results
            all_results.append({
                "model": model_path,
                "with_boxed_instructions": with_boxed_instructions,
                "accuracy": accuracy,
                "correct_count": count_correct,
                "total_count": count_total,
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "top_k": sampling_params.top_k,
            })

            print(f"\n\nModel: {model_path}")
            print(f"With boxed instructions: {with_boxed_instructions}")
            print(f"Temperature: {sampling_params.temperature}")
            print(f"Accuracy: {accuracy}")
            print("\n")

            # clear up memory
            del outputs
            del record

        # Move model cleanup to after all configurations are done
        del llm

    # Save aggregated results both as JSON and as a formatted table
    with open("aggregated_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Create a formatted table
    headers = ["Model", "Boxed Instr.", "Temp.", "Accuracy", "Correct", "Total"]
    table_data = [
        [
            r["model"],
            r["with_boxed_instructions"],
            r["temperature"],
            f"{r['accuracy']:.3f}",
            r["correct_count"],
            r["total_count"]
        ]
        for r in all_results
    ]
    
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    with open("aggregated_results.txt", "w") as f:
        f.write(table)

def main():
    for dataset in ["./datasets/trash_math_train_questions.json"]:
        print(f"Running dataset {dataset}...")
        main_dataset(dataset)

if __name__ == "__main__":
    main()
