from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch



# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Process User Input
user_prompt = "Write a python code to generate a bar chart of some sample data"
inputs = tokenizer.encode("Graph code generation: " + user_prompt, return_tensors="pt")

attention_mask = torch.ones_like(inputs)  # Create attention mask with 1s for non-pad tokens
outputs = model.generate(
    inputs,
    max_length=1000000,
    num_return_sequences=1,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id to eos_token_id
    attention_mask=attention_mask,  # Use the created attention mask
)

generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_code)