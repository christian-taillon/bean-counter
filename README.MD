# Bean Counter

Bean Counter is a Python script that analyzes text files using various tokenizers. It provides token, word, and character counts, as well as average tokens per word and characters per token.

## Setup Instructions

1. Ensure you have Python 3.7 or higher installed on your system.

2. Clone this repository or download the `bean_counter.py` file.

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the script:
   ```
   python bean_counter.py <file_path1> [file_path2] ...
   ```
   Replace `<file_path1>` with the path to the file you want to analyze. You can provide multiple file paths.

5. Follow the prompts to choose a tokenizer and optionally save the results to a file.

## Features

- Supports multiple tokenizers including GPT-4, GPT-3, Codex, GPT-2, OPT, GPT-NeoX, LLaMA-2, and BLOOM.
- Analyzes multiple files in one run.
- Provides token, word, and character counts.
- Calculates average tokens per word and characters per token.
- Option to save results to a file.

## Requirements

The script requires the following Python packages:

- tiktoken
- transformers

These can be installed using the provided `requirements.txt` file.

## Usage Example

```
python bean_counter.py sample1.txt sample2.txt
```

This will analyze both `sample1.txt` and `sample2.txt`, prompting you to choose a tokenizer and asking if you want to save the results to a file.

## Note

Some tokenizers may require additional setup or downloads when used for the first time. Ensure you have a stable internet connection when running the script with a new tokenizer.
