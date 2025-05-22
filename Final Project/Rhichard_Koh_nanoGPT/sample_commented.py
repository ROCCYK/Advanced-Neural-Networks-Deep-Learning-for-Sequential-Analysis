"""
Sample from a trained model
"""
import os  # Module for interacting with the operating system, e.g., file paths.
import pickle  # Module for serializing and deserializing Python objects.
from contextlib import nullcontext  # Null context manager for simpler condition handling.
import torch  # PyTorch library for deep learning.
import tiktoken  # Library for handling tokenization (encodings).
from model import GPTConfig, GPT  # Custom classes for configuring and creating the GPT model.

init_from = 'resume'  # Indicates how to initialize the model, either 'resume' or a GPT-2 variant (e.g., 'gpt2-xl').
out_dir = 'out'  # Directory to resume the model from (used if `init_from` is 'resume').
start = "\n"  # The starting prompt for generating text (can also point to a file with "FILE:filename").
num_samples = 10  # Number of samples to generate.
max_new_tokens = 500  # Maximum number of tokens to generate for each sample.
temperature = 0.8  # Sampling temperature; lower values mean less randomness, higher values mean more randomness.
top_k = 200  # Limit predictions to the top K tokens to reduce the probability of unlikely tokens.
seed = 1337  # Random seed for reproducibility.
device = 'cuda'  # Device to use for training and inference ('cpu', 'cuda', etc.).
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # Data type for computation.
compile = False  # Option to compile the model for faster inference (requires PyTorch 2.0).
exec(open('configurator.py').read())  # Execute `configurator.py` to override settings via command line or config file.

# Set random seed for reproducibility
torch.manual_seed(seed)  # Set the seed for PyTorch operations.
torch.cuda.manual_seed(seed)  # Set the seed for CUDA operations.
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 format on CUDA matmul for better performance.
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 format on cuDNN to improve efficiency.
device_type = 'cuda' if 'cuda' in device else 'cpu'  # Determine device type for later use in torch.autocast.
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]  # Set PyTorch data type.
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)  # Context for AMP (automatic mixed precision).

# Load the model
if init_from == 'resume':
    # Initialize from a saved checkpoint in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')  # Path to the checkpoint file.
    checkpoint = torch.load(ckpt_path, map_location=device)  # Load checkpoint from disk to the specified device.
    gptconf = GPTConfig(**checkpoint['model_args'])  # Load model configuration from checkpoint.
    model = GPT(gptconf)  # Create a GPT model based on configuration.
    state_dict = checkpoint['model']  # Extract the model weights.
    unwanted_prefix = '_orig_mod.'  # Prefix in state_dict keys to be removed.
    for k, v in list(state_dict.items()):  # Remove unwanted prefix from state dict keys.
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)  # Load the model weights into the model.
elif init_from.startswith('gpt2'):
    # Initialize from a pretrained GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))  # Load GPT-2 model with no dropout.

model.eval()  # Set the model to evaluation mode (disables dropout, etc.).
model.to(device)  # Move the model to the specified device.
if compile:
    model = torch.compile(model)  # Compile the model using PyTorch 2.0 (optional for better performance).

# Load meta information if available
load_meta = False  # Flag to check if meta.pkl should be loaded.
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:  # Check for older checkpoint configs.
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')  # Path to the meta information.
    load_meta = os.path.exists(meta_path)  # Check if the meta file exists.
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:  # Open the meta.pkl file in binary read mode.
        meta = pickle.load(f)  # Load the metadata.
    # Use the loaded metadata to encode/decode strings
    stoi, itos = meta['stoi'], meta['itos']  # Load the string-to-integer and integer-to-string dictionaries.
    encode = lambda s: [stoi[c] for c in s]  # Function to encode a string into a list of integers.
    decode = lambda l: ''.join([itos[i] for i in l])  # Function to decode a list of integers back to a string.
else:
    # Default to GPT-2 encoding if no meta information is found
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")  # Get the GPT-2 tokenizer.
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})  # Encode a string into a list of tokens.
    decode = lambda l: enc.decode(l)  # Decode a list of tokens back to a string.

# Encode the beginning of the prompt
if start.startswith('FILE:'):
    # If the start prompt is provided as a file, read the contents
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()  # Read the content of the file.
start_ids = encode(start)  # Encode the prompt into token IDs.
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]  # Convert prompt to tensor and add a batch dimension.

# Run text generation
with torch.no_grad():  # Disable gradient calculation for inference (more efficient).
    with ctx:  # Context manager for mixed precision if applicable.
        for k in range(num_samples):  # Generate `num_samples` of text.
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)  # Generate new tokens.
            print(decode(y[0].tolist()))  # Decode the generated tokens to text and print.
            print('---------------')  # Separator between samples.