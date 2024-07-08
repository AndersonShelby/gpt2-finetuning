# Crie um novo script para baixar o modelo pré-treinado
with open('download_model.py', 'w') as f:
    f.write("""
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Nome do modelo que queremos baixar
model_name = "gpt2"

# Diretório onde o modelo será salvo
save_directory = "models/gpt2-base/"

# Baixe o tokenizador e o modelo pré-treinado
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Salve o tokenizador e o modelo no diretório especificado
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"Modelo e tokenizador salvos em {save_directory}")
""")
