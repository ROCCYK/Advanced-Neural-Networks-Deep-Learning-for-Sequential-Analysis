"""
This training script can be run both on a single GPU in debug mode,
and also in a larger training run with Distributed Data Parallel (DDP).

Examples:
- Single GPU:
  $ python train.py --batch_size=32 --compile=False
- DDP with 4 GPUs on 1 node:
  $ torchrun --standalone --nproc_per_node=4 train.py
- DDP with 4 GPUs across 2 nodes (master and worker nodes specified):
  $ torchrun ... train.py
"""
import os
import time
import math
import pickle
from contextlib import nullcontext  # Provides a dummy context manager when no action is needed.

import numpy as np  # Used for numerical operations, including handling datasets.
import torch  # PyTorch for building and training models.
from torch.nn.parallel import DistributedDataParallel as DDP  # DDP for multi-GPU training.
from torch.distributed import init_process_group, destroy_process_group  # Manage distributed training processes.

from model import GPTConfig, GPT  # Import the model configuration and the GPT model.
import matplotlib.pyplot as plt  # Used for plotting loss graphs.

# -----------------------------------------------------------------------------
# Default configurations for training
out_dir = 'out'  # Output directory for checkpoints and logs.
eval_interval = 2000  # Interval for evaluating the model on validation data.
log_interval = 1  # Interval for printing training logs.
eval_iters = 200  # Number of iterations for evaluation.
eval_only = False  # If True, run evaluation and exit (no training).
always_save_checkpoint = True  # Save checkpoints after each evaluation regardless of performance.
init_from = 'scratch'  # Initialize model: 'scratch', 'resume', or a pre-trained model like 'gpt2*'.

# Weights and Biases (W&B) logging
wandb_log = False  # Whether to log training metrics to W&B.
wandb_project = 'owt'  # W&B project name.
wandb_run_name = 'gpt2'  # W&B run name.

# Data settings
dataset = 'openwebtext'  # Dataset to train on.
gradient_accumulation_steps = 5 * 8  # Steps to accumulate gradients before updating (simulates larger batch sizes).
batch_size = 12  # Mini-batch size for each GPU.
block_size = 1024  # Maximum context length used in training.

# Model settings
n_layer = 12  # Number of transformer layers.
n_head = 12  # Number of attention heads.
n_embd = 768  # Dimension of the embeddings.
dropout = 0.0  # Dropout rate (0 for pretraining, 0.1+ for finetuning).
bias = False  # Use bias in layer normalization and linear layers.

# Optimizer settings
learning_rate = 6e-4  # Maximum learning rate.
max_iters = 600000  # Total number of training iterations.
weight_decay = 1e-1  # Weight decay for regularization.
beta1 = 0.9  # Adam optimizer beta1.
beta2 = 0.95  # Adam optimizer beta2.
grad_clip = 1.0  # Clip gradients to this value to prevent exploding gradients.

# Learning rate decay settings
decay_lr = True  # Whether to decay the learning rate during training.
warmup_iters = 2000  # Number of warmup iterations.
lr_decay_iters = 600000  # Number of iterations for learning rate decay.
min_lr = 6e-5  # Minimum learning rate.

# DDP settings
backend = 'nccl'  # Backend for distributed training ('nccl' for NVIDIA GPUs).

# System settings
device = 'cuda'  # Device to use for training ('cpu', 'cuda', etc.).
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # Data type for training.
compile = True  # Whether to use PyTorch 2.0 compilation to optimize the model.
# -----------------------------------------------------------------------------
# Configuration loading
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # Override settings from command line or config file.
config = {k: globals()[k] for k in config_keys}  # Save configuration for logging.

# Initialize distributed data parallel (DDP) if applicable.
ddp = int(os.environ.get('RANK', -1)) != -1  # Check if running in DDP mode by seeing if 'RANK' environment variable exists and is not -1.
if ddp:
    # If in DDP mode, initialize the process group and set necessary variables.
    init_process_group(backend=backend)  # Initialize the process group for distributed training with the specified backend (e.g., 'nccl').
    ddp_rank = int(os.environ['RANK'])  # Get the global rank of the process from the environment variable.
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # Get the local rank within the current node.
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # Get the total number of processes involved in training.
    device = f'cuda:{ddp_local_rank}'  # Assign the GPU corresponding to the local rank to this process.
    torch.cuda.set_device(device)  # Set the CUDA device for PyTorch to use the assigned GPU.
    master_process = ddp_rank == 0  # Set master process flag to True if this is the rank 0 process, which will handle logging/checkpointing.
    seed_offset = ddp_rank  # Offset the random seed by the rank so that each process has a different random seed.
    # Adjust gradient accumulation steps to evenly distribute across multiple processes.
    assert gradient_accumulation_steps % ddp_world_size == 0  # Ensure gradient accumulation steps can be split evenly across processes.
    gradient_accumulation_steps //= ddp_world_size  # Reduce the accumulation steps by the world size to balance across processes.
else:
    # If not using DDP, set default values as if running on a single GPU or CPU.
    master_process = True  # This process is considered the master process, which handles logging/checkpointing.
    seed_offset = 0  # No seed offset needed for a single process.
    ddp_world_size = 1  # Only one process is used.

# Calculate tokens per iteration.
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size  # Calculate the number of tokens processed per iteration.
print(f"tokens per iteration will be: {tokens_per_iter:,}")  # Print the number of tokens per iteration in a human-readable format.

# Prepare output directory if this is the master process.
if master_process:
    os.makedirs(out_dir, exist_ok=True)  # Create the output directory if it doesn't exist, only done by the master process.

# Set random seed and configure CUDA settings.
torch.manual_seed(1337 + seed_offset)  # Set the random seed for reproducibility, offset by the process rank if using DDP.
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on matmul operations for faster computation (trades a little precision for speed).
torch.backends.cudnn.allow_tf32 = True  # Enable TF32 on cuDNN operations for better performance.
device_type = 'cuda' if 'cuda' in device else 'cpu'  # Determine whether the device is CUDA (GPU) or CPU.
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]  # Set the data type for training, based on specified dtype.
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)  # Set up a context manager for automatic mixed precision (AMP) if using GPU.

# Set up data directory path.
data_dir = os.path.join('data', dataset)  # Define the directory where dataset files are stored.

# Function to load a batch of data.
def get_batch(split):
    """Load a batch of data for training or validation."""
    # Create a memory-mapped object for reading the binary data file.
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')  # Load data from train or validation split.
    ix = torch.randint(len(data) - block_size, (batch_size,))  # Randomly select start indices for the batch, ensuring there's enough room for the block size.
    
    # Create input (x) and target (y) tensors.
    # `x` is a tensor of input tokens, `y` is the next token for each position in `x`.
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])  # Stack each batch sequence for the inputs.
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])  # Stack each batch sequence for the targets (offset by 1).

    # Move data to the appropriate device.
    if device_type == 'cuda':
        # Pin memory for faster transfer to GPU and move data to the specified GPU device.
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y  # Return the input and target tensors.

# Initialize training settings.
iter_num = 0  # Initialize iteration counter.
best_val_loss = 1e9  # Set initial validation loss to a very large value.

# Load meta information if available (e.g., vocabulary size).
meta_path = os.path.join(data_dir, 'meta.pkl')  # Path to meta information file.
meta_vocab_size = None
if os.path.exists(meta_path):  # If the meta file exists, load it.
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']  # Get vocabulary size from the meta file.
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Model initialization
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=None, dropout=dropout)
if init_from == 'scratch':
    # Initialize a new model from scratch.
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size else 50304  # Set vocabulary size to meta_vocab_size or default GPT-2 vocab size.
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    # Resume training from a saved checkpoint.
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')  # Path to checkpoint file.
    checkpoint = torch.load(ckpt_path, map_location=device)  # Load checkpoint.
    checkpoint_model_args = checkpoint['model_args']  # Extract model arguments from checkpoint.
    # Update model arguments based on the checkpoint.
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)  # Create the model with updated arguments.
    state_dict = checkpoint['model']  # Extract model weights.
    # Remove unwanted prefix in checkpoint keys.
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)  # Load model weights.
    iter_num = checkpoint['iter_num']  # Resume iteration number.
    best_val_loss = checkpoint['best_val_loss']  # Load the best validation loss.
elif init_from.startswith('gpt2'):
    # Initialize from pre-trained GPT-2 weights.
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)  # Load GPT-2 weights with optional overrides.
    # Extract model arguments from the loaded configuration.
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# Crop the model's block size if necessary.
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size  # Update model argument for checkpoint consistency.
model.to(device)  # Move model to the specified device.

# Initialize gradient scaler for mixed precision.
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Configure optimizer.
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])  # Load optimizer state if resuming training.
checkpoint = None  # Free up memory by deleting the checkpoint.

# Compile model if applicable.
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # Compile the model to optimize performance (requires PyTorch 2.0).

# Wrap model in DDP container if applicable.
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])  # Wrap the model with DistributedDataParallel if using DDP.

# Function to estimate loss for training and validation splits.
@torch.no_grad()  # Disable gradient calculation to speed up evaluation and reduce memory usage.
def estimate_loss():
    out = {}
    model.eval()  # Set model to evaluation mode.
    for split in ['train', 'val']:  # Loop over training and validation splits.
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)  # Get a batch of data.
            with ctx:
                logits, loss = model(X, Y)  # Forward pass to calculate loss.
            losses[k] = loss.item()  # Store the loss value.
        out[split] = losses.mean()  # Calculate the mean loss for the split.
    model.train()  # Set model back to training mode.
    return out

# Learning rate scheduler (cosine decay with warmup).
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters  # Linear warmup.
    if it > lr_decay_iters:
        return min_lr  # Minimum learning rate after decay period.
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)  # Calculate decay ratio.
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine decay coefficient.
    return min_lr + coeff * (learning_rate - min_lr)  # Calculate decayed learning rate.

# Initialize Weights & Biases (W&B) logging if applicable.
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)  # Initialize W&B logging.

# Training loop initialization.
X, Y = get_batch('train')  # Get the first batch of training data.
t0 = time.time()  # Record the start time.
local_iter_num = 0  # Local iteration counter for this process.
raw_model = model.module if ddp else model  # If using DDP, unwrap the model.
running_mfu = -1.0  # Initialize moving average of model utilization (mfu).
train_losses_list = []  # List to store training losses.
val_losses_list = []  # List to store validation losses.
epoch_list = []  # List to store iteration numbers.

# Training loop.
while True:
    # Set learning rate for the current iteration.
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # Update learning rate for each parameter group.

    # Evaluate the model and save checkpoints.
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()  # Estimate training and validation losses.
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Store loss metrics for plotting.
        train_losses_list.append(losses['train'])
        val_losses_list.append(losses['val'])
        epoch_list.append(iter_num)

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,  # Convert MFU to percentage.
            })

        # Save checkpoint if validation loss improves or if always_save_checkpoint is True.
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))  # Save checkpoint.

    if iter_num == 0 and eval_only:
        break  # If only evaluation is needed, exit after the first evaluation.

    # Forward and backward pass, with gradient accumulation.
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)  # Sync gradients only in the last micro-step.
        with ctx:
            logits, loss = model(X, Y)  # Forward pass.
            loss = loss / gradient_accumulation_steps  # Scale the loss to account for gradient accumulation.
        X, Y = get_batch('train')  # Prefetch the next batch.
        scaler.scale(loss).backward()  # Backward pass with gradient scaling.

    # Clip gradients to prevent gradient explosion.
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)  # Unscale the gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # Clip gradients.

    # Step the optimizer and update the gradient scaler.
    scaler.step(optimizer)  # Perform an optimization step.
    scaler.update()  # Update the gradient scaler.
    optimizer.zero_grad(set_to_none=True)  # Clear gradients.

    # Timing and logging.
    t1 = time.time()
    dt = t1 - t0  # Calculate duration since last iteration.
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps  # Calculate loss for logging.
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)  # Estimate model utilization.
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu  # Update running MFU.
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

    iter_num += 1  # Increment iteration counter.
    local_iter_num += 1

    # Termination condition.
    if iter_num > max_iters:
        break  # Exit the training loop if the maximum iterations are reached.

if ddp:
    destroy_process_group()  # Destroy the process group if using DDP.

# Plot training and validation loss over time.
plt.figure(figsize=(16, 9))
plt.plot(epoch_list, train_losses_list, label="Train Loss", marker="o")
plt.plot(epoch_list, val_losses_list, label="Val Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Val Loss with Custom Hyperparameters")
plt.legend()
plt.grid(True)
plt.savefig("loss plot.png")  # Save the loss plot to a file.
