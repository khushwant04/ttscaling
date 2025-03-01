# Test-Time Scaling

This repository demonstrates a test-time scaling approach for large language models using a LLaMA variant from Hugging Face. In this example, we generate multiple candidate responses for a given prompt and then aggregate them by selecting the candidate with the highest average log probability. This technique can help reduce uncertainty and improve robustness at inference time.

## Overview

Test-time scaling is a model‑agnostic technique that leverages additional compute during inference to generate multiple outputs. By aggregating these outputs (for example, selecting the one with the best statistical properties), you can reduce variance and obtain a more robust final prediction without changing the underlying model.

In this repository, we:
- Use the Hugging Face Transformers library to load a LLaMA 3.2-3B-Instruct model.
- Generate 5 candidate outputs for a prompt using stochastic sampling.
- Compute the average log probability for each candidate and select the best one.

## Features

- **Multiple Candidate Generation:** Generates multiple outputs using sampling.
- **Log Probability Aggregation:** Computes per-token log probabilities and averages them over each candidate.
- **Best Candidate Selection:** Picks the candidate with the highest average log probability as the final aggregated answer.
- **Easy-to-Use Code:** Provided as a standalone Python script that can be run with a single command.

## Requirements

- Python 3.8 or later
- [PyTorch](https://pytorch.org/)
- [Transformers](https://github.com/huggingface/transformers)
- Access to the gated model: `meta-llama/Llama-3.2-3B-Instruct` (ensure you have the proper credentials)

## Installation

1. **Clone this repository:**

   ```bash
   git clone https://github.com/your-username/test-time-scaling-llama.git
   cd test-time-scaling-llama
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install torch transformers
   ```

## Usage

1. **Ensure you have access to the LLaMA model:**  
   Make sure you have authenticated with Hugging Face (using `huggingface-cli login` if needed).

2. **Run the Python script:**

   ```bash
   python test_time_scaling.py
   ```

   This will generate multiple candidate outputs for the prompt "Explain the significance of AI in 2025." and print both all generated outputs and the final aggregated output (i.e. the candidate with the highest average log probability).

## Code Explanation

Below is a brief explanation of the key sections in the script:

- **Model and Tokenizer Loading:**  
  Loads the model `meta-llama/Llama-3.2-3B-Instruct` and its corresponding tokenizer with proper settings (including `trust_remote_code=True`).

- **Prompt Tokenization:**  
  The provided prompt is tokenized and moved to the same device as the model.

- **Generation with Output Scores:**  
  The `generate()` method is called with sampling enabled and outputs are returned as a dictionary containing both generated sequences and scores for each step.

- **Log Probability Computation:**  
  For each candidate, the code computes the log softmax of the output logits to get per-token log probabilities. It then averages these probabilities across the generated tokens (excluding the input prompt).

- **Candidate Selection:**  
  The candidate with the highest average log probability is selected as the final output.

## Contributing

Contributions, suggestions, or improvements to the code are very welcome! Please feel free to open an issue or submit a pull request.

## License

This repository is provided for demonstration purposes. Make sure to follow the license terms of the underlying model (Meta LLaMA) as specified in the model card on Hugging Face.