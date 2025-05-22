import os  # Module for interacting with the operating system, e.g., file paths.
import pickle  # Module for serializing and deserializing Python objects.
import requests  # Module to make HTTP requests, such as downloading files.
import numpy as np  # Import numpy and use the alias `np` for numerical operations.

# Download the Taylor Swift Lyrics dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'lyrics.txt')  # Construct the file path for the dataset.
if not os.path.exists(input_file_path):  # Check if the dataset file already exists.
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'  # URL of the dataset.
    with open(input_file_path, 'w') as f:  # Open the file for writing.
        f.write(requests.get(data_url).text)  # Download the dataset from the URL and save it to the file.

# Load the dataset from the file
with open(input_file_path, 'r', encoding='utf-8') as f:  # Open the file in read mode with UTF-8 encoding.
    data = f.read()  # Read the entire content of the file into the `data` variable.
print(f"length of dataset in characters: {len(data):,}")  # Print the length of the dataset in characters.

# Get all the unique characters that occur in this text
chars = sorted(list(set(data)))  # Create a sorted list of all unique characters in the dataset.
vocab_size = len(chars)  # Determine the number of unique characters (vocabulary size).
print("all the unique characters:", ''.join(chars))  # Print all unique characters as a string.
print(f"vocab size: {vocab_size:,}")  # Print the size of the vocabulary.

# Create a mapping from characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}  # Dictionary that maps each character to a unique integer.
itos = {i: ch for i, ch in enumerate(chars)}  # Dictionary that maps each integer back to the corresponding character.

# Encoder: function to encode a string as a list of integers
def encode(s):
    return [stoi[c] for c in s]  # Convert each character in the string `s` to its corresponding integer.

# Decoder: function to decode a list of integers back to a string
def decode(l):
    return ''.join([itos[i] for i in l])  # Convert each integer in the list `l` back to its corresponding character.

# Create the train and validation splits
n = len(data)  # Determine the total length of the dataset.
train_data = data[:int(n * 0.9)]  # Take 90% of the data for training.
val_data = data[int(n * 0.9):]  # Take the remaining 10% of the data for validation.

# Encode both splits to lists of integers
train_ids = encode(train_data)  # Encode the training data to a list of integers.
val_ids = encode(val_data)  # Encode the validation data to a list of integers.
print(f"train has {len(train_ids):,} tokens")  # Print the number of tokens in the training data.
print(f"val has {len(val_ids):,} tokens")  # Print the number of tokens in the validation data.

# Export the encoded data to binary files
train_ids = np.array(train_ids, dtype=np.uint16)  # Convert the training data list to a NumPy array of type uint16.
val_ids = np.array(val_ids, dtype=np.uint16)  # Convert the validation data list to a NumPy array of type uint16.
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))  # Save the training data to a binary file.
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))  # Save the validation data to a binary file.

# Save the metadata information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,  # Store the vocabulary size.
    'itos': itos,  # Store the character-to-integer mapping.
    'stoi': stoi,  # Store the integer-to-character mapping.
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:  # Open a file in binary write mode to save metadata.
    pickle.dump(meta, f)  # Serialize and save the metadata dictionary using pickle.