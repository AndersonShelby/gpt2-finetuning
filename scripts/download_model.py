from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = "gpt2"
save_directory = "models/gpt2-base/"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"Modelo e tokenizador salvos em {save_directory}")
