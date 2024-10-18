import tiktoken
import sys
from transformers import AutoTokenizer

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
    if tokenizer_type == "tiktoken":
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))
    elif tokenizer_type == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(encoding_name)
        return len(tokenizer.encode(string))

def analyze_file(file_path: str, encoding_name: str, tokenizer_type: str) -> None:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        token_count = num_tokens_from_string(content, encoding_name, tokenizer_type)
        word_count = len(content.split())
        char_count = len(content)
        
        avg_tokens_per_word = token_count / word_count if word_count > 0 else 0
        avg_chars_per_token = char_count / token_count if token_count > 0 else 0

        print(f"\nFile: {file_path}")
        print(f"Tokenizer: {encoding_name}")
        print(f"Total tokens: {token_count}")
        print(f"Total words: {word_count}")
        print(f"Total characters: {char_count}")
        print(f"Average tokens per word: {avg_tokens_per_word:.2f}")
        print(f"Average characters per token: {avg_chars_per_token:.2f}")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
    else:
        file_path = sys.argv[1]
        encoding_name, _, tokenizer_type = get_tokenizer_choice()
        analyze_file(file_path, encoding_name, tokenizer_type)
