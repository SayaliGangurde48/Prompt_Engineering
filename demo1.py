import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load GPT-2 model
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Accept prompt
prompt = input("Enter your prompt: ")

inputs = tokenizer(prompt, return_tensors="pt")

# Generate response
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True
    )

generated_ids = output.sequences[0][inputs.input_ids.shape[1]:]

print("\nGenerated Response:")
print(tokenizer.decode(generated_ids, skip_special_tokens=True))

print("\nToken Analysis")
print("-" * 90)
print("{:<5}{:<15}{:<15}{:<15}{:<20}{}".format(
    "No.", "Token", "Log Prob", "Probability", "Cumulative Prob", "Color"
))

cumulative = 1.0

for i, (token_id, score) in enumerate(zip(generated_ids, output.scores), start=1):

    log_probs = torch.log_softmax(score, dim=-1)

    log_prob = log_probs[0, token_id].item()
    probability = math.exp(log_prob)

    cumulative *= probability

    # Color Coding
    if probability >= 0.80:
        color = "🟢 Green"
    elif probability >= 0.50:
        
        
    
        color = "🟡 Yellow"
    else:
        color = "🔴 Red"

    token = tokenizer.decode([token_id])

    print("{:<5}{:<15}{:<15.4f}{:<15.4f}{:<20.8f}{}".format(
        i,
        repr(token),
        log_prob,
        probability,
        cumulative,
        color
    ))