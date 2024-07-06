import json
import os

def process_raw_data(input_file, output_file):
    # Verifica se o diretório de saída existe, caso contrário, cria-o
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        raw_data = json.load(infile)
        for chat in raw_data:
            conversation = f"Agent: {chat['agent']}\nCustomer: {chat['customer']}\n"
            outfile.write(conversation + "\n")

if __name__ == "__main__":
    input_path = "data/raw/support_chats.json"
    output_path = "data/processed/support_chats_cleaned.txt"
    process_raw_data(input_path, output_path)
