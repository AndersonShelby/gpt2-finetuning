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
