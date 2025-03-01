import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model Initlization
model_name = "meta-llama/Llama-3.2-3B-Instruct"

# Load tokenizer and model with appropriate trust_remote_code if needed.
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Define your prompt
prompt = "Explain the significance of AI in 2025."

# Tokenize prompt and move to the model's device
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate multiple outputs with sampling and with output scores enabled.
# We set return_dict_in_generate=True and output_scores=True so we can access the logit scores.
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=100,
    num_return_sequences=5,
    do_sample=True,         # enables stochastic sampling
    temperature=0.7,        # adjust sampling randomness
    return_dict_in_generate=True,
    output_scores=True
)

# The generated sequences include the input tokens. 
# Let input_length be the number of tokens in the prompt.
input_length = inputs["input_ids"].shape[1]
sequences = outputs.sequences  # Shape: [num_return_sequences, total_length]

# The outputs.scores is a list (one per generated token) where each element has shape: [num_return_sequences, vocab_size]
# We'll compute the log probability for each generated token in each candidate.
log_probs = []  # List to store per-token log probabilities per candidate

for i, step_scores in enumerate(outputs.scores):
    # Compute log softmax to obtain log probabilities for this generation step.
    step_log_probs = F.log_softmax(step_scores, dim=-1)  # Shape: [num_return_sequences, vocab_size]
    
    # The generated tokens start at index input_length.
    # For the i-th generated token, get its token id from each candidate.
    token_ids = sequences[:, input_length + i]  # Shape: [num_return_sequences]
    
    # Gather the log probability for the token id for each candidate.
    token_log_prob = step_log_probs.gather(dim=1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    log_probs.append(token_log_prob)

# Stack the list to form a tensor of shape: [num_return_sequences, max_new_tokens]
log_probs_tensor = torch.stack(log_probs, dim=1)

# Compute the average log probability for each candidate.
avg_log_probs = log_probs_tensor.mean(dim=1)  # Shape: [num_return_sequences]

# Select the candidate with the highest average log probability.
best_idx = avg_log_probs.argmax().item()
final_output = tokenizer.decode(sequences[best_idx], skip_special_tokens=True)

# Print all candidate responses and the selected final output.
print("Generated outputs:")
for idx, seq in enumerate(sequences):
    decoded = tokenizer.decode(seq, skip_special_tokens=True)
    print(f"Output {idx+1}:")
    print(decoded)
    print("-" * 40)

print("\nFinal aggregated output (best candidate):")
print(final_output)
