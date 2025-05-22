import torch._dynamo  # Import the torch._dynamo module, which is an experimental feature in PyTorch to help with optimization.
torch._dynamo.config.suppress_errors = True  # Suppress errors from torch._dynamo to avoid interruptions during experimentation.

# Train a miniature character-level Taylor Swift Lyrics model

out_dir = 'out-ts-char'  # Directory to save the output, such as checkpoints and logs.
eval_interval = 250  # How often (in iterations) to evaluate the model on validation data.
eval_iters = 200  # Number of iterations to run during evaluation.
log_interval = 10  # How often to print training logs (e.g., loss values) during training.

# We expect to overfit on this small dataset, so only save checkpoints when validation improves.
always_save_checkpoint = False  # Only save the model checkpoint if validation loss improves.

wandb_log = False  # Whether to log metrics to Weights and Biases (W&B); can be overridden via command line.
wandb_project = 'ts-char'  # The project name in W&B.
wandb_run_name = 'mini-gpt'  # The run name for W&B.

dataset = 'ts_char'  # Dataset identifier; here, we’re using the Taylor Swift Lyrics dataset (`ts_char`).
gradient_accumulation_steps = 1  # Number of gradient accumulation steps before updating weights.
batch_size = 64  # The number of examples per training batch.
block_size = 512  # Context window size (how many previous characters to consider when predicting the next one).

# Model architecture - a smaller version of GPT
n_layer = 8  # Number of transformer layers in the model.
n_head = 8  # Number of attention heads in each layer.
n_embd = 512  # Size of the embedding (hidden) dimension.
dropout = 0.3  # Dropout rate to prevent overfitting during training.

learning_rate = 1e-4  # Learning rate for the optimizer; can go higher because it's a small model.
max_iters = 10000  # Maximum number of iterations to train the model.
lr_decay_iters = 10000  # Number of iterations over which the learning rate will decay; typically equal to max_iters.
min_lr = 1e-5  # Minimum learning rate after decay (usually set to learning_rate / 10).
beta2 = 0.88  # The beta2 parameter for the Adam optimizer, adjusted for smaller batch sizes.

warmup_iters = 100  # Number of warmup iterations where the learning rate linearly increases to its initial value.

# Optional settings for running on low-power devices like a MacBook
# device = 'cpu'  # Uncomment this to run on the CPU only (instead of GPU).
# compile = False # Uncomment this to disable model compilation for efficiency.
