# Treinamento de Modelo GPT-2

Este projeto tem como objetivo treinar um modelo GPT-2 baseado em dados de conversas de suporte ao cliente. Utilizamos a biblioteca `transformers` da Hugging Face para realizar o treinamento e o ajuste fino do modelo.

## Estrutura do Diretório

```
meu-repositorio/
├── data/
│   ├── raw/
│   │   └── support_chats.json
│   ├── processed/
│   │   └── processed_chats.json
├── models/
│   └── gpt2-base/
├── notebooks/
├── scripts/
│   ├── preprocess_data.py
│   └── train_model.py
└── README.md
```

## Instalação

1. **Clone o repositório:**

   ```bash
   git clone https://github.com/seu-usuario/meu-repositorio.git
   cd meu-repositorio
   ```

2. **Instale as dependências:**

   ```bash
   pip install transformers datasets
   ```

## Estrutura dos Dados

Os dados de treinamento devem estar no formato JSON e armazenados no diretório `data/raw/`. Um exemplo de arquivo `support_chats.json` é fornecido abaixo:

```json
[
  {
    "timestamp": "2024-07-06T12:34:56",
    "agent": "Olá! Como posso ajudar hoje?",
    "customer": "Meu pedido não chegou ainda.",
    "agent": "Qual é o número do seu pedido?",
    "customer": "É 12345.",
    "agent": "Deixe-me verificar... Seu pedido está a caminho e deve chegar amanhã."
  },
  ...
]
```

## Pré-processamento dos Dados

Utilize o script `preprocess_data.py` para processar os dados brutos e prepará-los para o treinamento.

```python
import json

def preprocess_data(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    processed_data = []
    for chat in data:
        processed_chat = {
            "timestamp": chat["timestamp"],
            "text": f'Agent: {chat["agent"]} Customer: {chat["customer"]}'
        }
        processed_data.append(processed_chat)

    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    input_path = "data/raw/support_chats.json"
    output_path = "data/processed/processed_chats.json"
    preprocess_data(input_path, output_path)
```

Execute o script:

```bash
python scripts/preprocess_data.py
```

## Download do Modelo GPT-2 Base

Use o seguinte script para baixar o modelo GPT-2 base:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = "gpt2"
save_directory = "models/gpt2-base/"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"Modelo e tokenizador salvos em {save_directory}")
```

## Treinamento do Modelo

Utilize o script `train_model.py` para treinar o modelo:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def train_gpt2_model(dataset_path, model_output_dir):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    def load_dataset(file_path, tokenizer, block_size=128):
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=block_size,
            overwrite_cache=True
        )

    def create_data_collator(tokenizer):
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

    dataset = load_dataset(dataset_path, tokenizer)
    data_collator = create_data_collator(tokenizer)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(model_output_dir)

if __name__ == "__main__":
    dataset_path = "data/processed/processed_chats.json"
    model_output_dir = "models/gpt2-base/"
    train_gpt2_model(dataset_path, model_output_dir)
```

Execute o script:

```bash
python scripts/train_model.py
```

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests para melhorias.
