import tiktoken
import sys
import os
from transformers import AutoTokenizer
from tqdm import tqdm

TOKENIZERS = {
    1: ("cl100k_base", "GPT-4 (default)", "tiktoken"),
    2: ("p50k_base", "GPT-3", "tiktoken"),
    3: ("r50k_base", "Codex", "tiktoken"),
    4: ("gpt2", "GPT-2", "tiktoken"),
    5: ("facebook/opt-350m", "OPT", "huggingface"),
    6: ("EleutherAI/gpt-neox-20b", "GPT-NeoX", "huggingface"),
    7: ("meta-llama/Llama-2-7b-hf", "LLaMA-2", "huggingface"),
    8: ("bigscience/bloom", "BLOOM", "huggingface")
}

def get_tokenizer_choice():
    print("Choose a tokenizer:")
    for key, (_, name, _) in TOKENIZERS.items():
        print(f"{key}. {name}")
    
    while True:
        try:
            choice = int(input("Enter your choice (1-8), or press Enter for default: ") or 1)
            if 1 <= choice <= 8:
                return TOKENIZERS[choice]
            else:
                print("Please enter a number between 1 and 8.")
        except ValueError:
            print("Please enter a valid number.")

def num_tokens_from_string(string: str, encoding_name: str, tokenizer_type: str) -> int:
    try:
        if tokenizer_type == "tiktoken":
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(string))
        elif tokenizer_type == "huggingface":
            tokenizer = AutoTokenizer.from_pretrained(encoding_name, use_auth_token=True)
            return len(tokenizer.encode(string))
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        print("Falling back to character count as an approximation.")
        return len(string)

def save_results(results: str, output_file: str) -> None:
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(results)
    print(f"Results saved to {output_file}")

def analyze_file(file_path: str, encoding_name: str, tokenizer_type: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        print("Tokenizing...")
        token_count = num_tokens_from_string(content, encoding_name, tokenizer_type)
        
        print("Analyzing...")
        word_count = len(content.split())
        char_count = len(content)
        
        avg_tokens_per_word = token_count / word_count if word_count > 0 else 0
        avg_chars_per_token = char_count / token_count if token_count > 0 else 0

        result = f"\nFile: {file_path}\n"
        result += f"Tokenizer: {encoding_name}\n"
        result += f"Total tokens: {token_count}\n"
        result += f"Total words: {word_count}\n"
        result += f"Total characters: {char_count}\n"
        result += f"Average tokens per word: {avg_tokens_per_word:.2f}\n"
        result += f"Average characters per token: {avg_chars_per_token:.2f}\n"
        
        print(result)
        return result
        
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found.\n"
    except Exception as e:
        return f"An error occurred: {str(e)}\n"

def analyze_files(file_paths: list, encoding_name: str, tokenizer_type: str) -> str:
    results = ""
    for file_path in file_paths:
        results += analyze_file(file_path, encoding_name, tokenizer_type)
        results += "\n" + "-"*50 + "\n"
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path1> [file_path2] ...")
    else:
        file_paths = sys.argv[1:]
        encoding_name, model_name, tokenizer_type = get_tokenizer_choice()
        
        if encoding_name == "meta-llama/Llama-2-7b-hf":
            print("Note: LLaMA-2 tokenizer requires special access. If you don't have access, the script will fall back to character count.")
        
        results = analyze_files(file_paths, encoding_name, tokenizer_type)
        
        save_option = input("Do you want to save the results to a file? (y/n): ").lower()
        if save_option == 'y':
            output_file = input("Enter the output file name: ")
            save_results(results, output_file)
