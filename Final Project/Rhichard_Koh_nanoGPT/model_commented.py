"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI.
2) huggingface/transformers PyTorch implementation.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Custom Layer Normalization Class
class LayerNorm(nn.Module):
    """Layer normalization with an optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))  # Scale parameter for normalization.
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # Bias parameter if bias is True.

    def forward(self, input):
        # Apply layer normalization with learned scale (weight) and optional bias.
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# Self-attention mechanism used in transformer models.
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # Ensure embedding dimension is divisible by number of heads.
        # Linear layers to compute key, query, and value vectors.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Linear layer to project the output.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Dropout layers for regularization.
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Attention parameters.
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # Use flash attention if available (faster and more efficient).
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Create a causal mask to ensure tokens attend only to previous tokens.
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension.

        # Compute query, key, value for all heads in the batch.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # Reshape for multi-head attention.
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Apply causal self-attention.
        if self.flash:
            # Use efficient attention mechanism.
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # Manual implementation of attention (less efficient).
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # Apply causal mask.
            att = F.softmax(att, dim=-1)  # Softmax to normalize attention scores.
            att = self.attn_dropout(att)
            y = att @ v  # Multiply attention weights with values.
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Concatenate heads.

        # Output projection.
        y = self.resid_dropout(self.c_proj(y))
        return y

# Feedforward network used after attention in each transformer block.
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)  # Linear layer to expand dimensionality.
        self.gelu = nn.GELU()  # GELU activation function.
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)  # Project back to original dimension.
        self.dropout = nn.Dropout(config.dropout)  # Dropout layer for regularization.

    def forward(self, x):
        x = self.c_fc(x)  # First linear transformation.
        x = self.gelu(x)  # Apply activation function.
        x = self.c_proj(x)  # Second linear transformation.
        x = self.dropout(x)  # Apply dropout.
        return x

# Transformer block consisting of attention and MLP with layer normalization.
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)  # First layer normalization.
        self.attn = CausalSelfAttention(config)  # Self-attention mechanism.
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)  # Second layer normalization.
        self.mlp = MLP(config)  # Feedforward network.

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # Apply layer norm, then attention, then add residual.
        x = x + self.mlp(self.ln_2(x))  # Apply layer norm, then MLP, then add residual.
        return x

# Configuration class for GPT model.
@dataclass
class GPTConfig:
    block_size: int = 1024  # Maximum sequence length.
    vocab_size: int = 50304  # Vocabulary size (GPT-2 default rounded up).
    n_layer: int = 12  # Number of transformer layers.
    n_head: int = 12  # Number of attention heads.
    n_embd: int = 768  # Dimension of embeddings.
    dropout: float = 0.0  # Dropout rate.
    bias: bool = True  # Whether to use bias in layers.

# GPT Model class.
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Transformer model components.
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # Token embedding.
            wpe=nn.Embedding(config.block_size, config.n_embd),  # Positional embedding.
            drop=nn.Dropout(config.dropout),  # Dropout layer.
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer blocks.
            ln_f=LayerNorm(config.n_embd, bias=config.bias),  # Final layer normalization.
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # Output layer (language modeling head).

        # Weight tying for efficiency.
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all model weights.
        self.apply(self._init_weights)
        # Special initialization for residual projections as per GPT-2 paper.
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Print number of parameters in the model.
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model, excluding position embeddings if requested."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize weights in the model."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Initialize linear layers.
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # Initialize biases to zero.
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Initialize embedding layers.

    def forward(self, idx, targets=None):
        """Forward pass of the model."""
        device = idx.device
        b, t = idx.size()  # Batch size and sequence length.
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # Positional indices.

        # Embedding layers.
        tok_emb = self.transformer.wte(idx)  # Token embeddings.
        pos_emb = self.transformer.wpe(pos)  # Position embeddings.
        x = self.transformer.drop(tok_emb + pos_emb)  # Add embeddings and apply dropout.

        # Pass through transformer blocks.
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # Final layer normalization.

        if targets is not None:
            # Calculate loss if targets are provided.
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # For inference, only output logits for the last position.
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """Crop the model's block size (sequence length) if necessary."""
        # Ensure the new block size is not larger than the original block size.
        assert block_size <= self.config.block_size

        # Update the model's configuration with the new, smaller block size.
        self.config.block_size = block_size

        # Update the positional embeddings (`wpe`) to reflect the smaller block size.
        # This crops the positional embedding weights to only include embeddings for positions up to `block_size`.
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

        # Iterate through all the transformer blocks and adjust their attention masks.
        for block in self.transformer.h:
            # Check if the block's attention mechanism has a bias attribute.
            if hasattr(block.attn, 'bias'):
                # Crop the causal mask (`bias`) used in the attention mechanism to the new block size.
                # This ensures that the causal mask only covers the reduced block size.
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """Load model weights from a pretrained GPT-2 model."""
        # Ensure the provided model type is valid.
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # Set override_args to an empty dictionary if none provided.
        override_args = override_args or {}
        # Ensure only 'dropout' can be overridden.
        assert all(k == 'dropout' for k in override_args)

        # Import the pretrained model from HuggingFace Transformers.
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")

        # Load the default configuration settings for the specified model type.
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # Small GPT-2 model (124M params).
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # Medium GPT-2 model (350M params).
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # Large GPT-2 model (774M params).
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # GPT-2 XL model (1558M params).
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        # Set vocabulary size, block size, and bias (GPT-2 specific settings).
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True

        # Override the dropout rate if specified in `override_args`.
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        # Create a GPT model instance with the provided configuration.
        config = GPTConfig(**config_args)
        model = GPT(config)

        # Load the pretrained model from HuggingFace's implementation.
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()  # Get state dictionary of HuggingFace model.

        # Copy parameters from HuggingFace model to this model.
        # Some weights need to be transposed (due to how Conv1D weights are represented in GPT-2).
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k in model.state_dict().keys():
            if any(k.endswith(w) for w in transposed):
                # If the key ends with one of the transposed names, transpose it when copying.
                with torch.no_grad():
                    model.state_dict()[k].copy_(sd_hf[k].t())
            else:
                # Otherwise, copy weights directly.
                with torch.no_grad():
                    model.state_dict()[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure the optimizer with specified hyperparameters."""
        # Get all parameters that require gradients.
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Separate parameters into those that will be decayed and those that will not.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]  # Weights (matrices) will be decayed.
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]  # Biases and LayerNorms will not be decayed.
        
        # Create parameter groups with and without weight decay.
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Check if the fused version of AdamW is available and should be used.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        
        # Instantiate the AdamW optimizer.
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model FLOPs utilization (MFU) compared to A100 peak performance."""
        # Get the total number of parameters in the model.
        N = self.get_num_params()
        # Extract model configuration values.
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        
        # Calculate the number of FLOPs per token.
        flops_per_token = 6 * N + 12 * L * H * Q * T
        # Calculate the number of FLOPs per forward-backward pass.
        flops_per_fwdbwd = flops_per_token * T
        # Calculate the number of FLOPs per iteration.
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        # Calculate the achieved FLOPs per second.
        flops_achieved = flops_per_iter * (1.0 / dt)
        # Theoretical peak FLOPs of an A100 GPU (in TFLOPS).
        flops_promised = 312e12
        # Calculate Model FLOPs Utilization (MFU).
        mfu = flops_achieved / flops_promised
        
        return mfu

    @torch.no_grad()  # Disable gradient calculation to reduce memory usage during inference.
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text by sampling from the model's output logits."""
        for _ in range(max_new_tokens):
            # If the sequence is longer than the model's block size, crop it to the last block_size tokens.
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass to get the logits for the current sequence.
            logits, _ = self(idx_cond)
            # Extract the logits for the last position and scale by the temperature.
            logits = logits[:, -1, :] / temperature
            
            # If top_k is specified, only keep the top_k most likely tokens.
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Convert logits to probabilities using softmax.
            probs = F.softmax(logits, dim=-1)
            
            # Sample the next token from the probability distribution.
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append the sampled token to the running sequence.
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx