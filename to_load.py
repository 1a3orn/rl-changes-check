from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

to_load = [
    ("agentica-org/DeepScale-R-1.5B-Preview", "agentica"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseek"),
    ("Qwen/Qwen2.5-Math-1.5B", "qwen"),
]


def main():
    # Load and save each model
    for model_id, save_path in to_load:
        print(f"Loading {model_id}...")
    
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Save model and tokenizer
        print(f"Saving to {save_path}...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        # Clear from memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

        print(f"Completed {model_id}\n")


if __name__ == "__main__":
    main()
