from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import json
import torch

class CustomDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=128):
        self.examples = []

        with open(file_path, 'r') as f:
            data = json.load(f)

        for item in data:
            tokenized_example = tokenizer(
                item["text"], 
                max_length=block_size, 
                padding="max_length", 
                truncation=True,
                return_tensors="pt"
            )
            self.examples.append(tokenized_example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {key: val.squeeze(0) for key, val in item.items()}

def train_gpt2_model(dataset_path, model_output_dir):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    dataset = CustomDataset(tokenizer, dataset_path)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    model.config.pad_token_id = model.config.eos_token_id
    trainer.train()
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

if __name__ == "__main__":
    dataset_path = "data/processed/processed_chats.json"
    model_output_dir = "models/gpt2-base/"
    train_gpt2_model(dataset_path, model_output_dir)
