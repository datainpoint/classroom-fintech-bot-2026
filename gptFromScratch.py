# Download Friends_Transcript.txt to working directory
#!wget https://raw.githubusercontent.com/datainpoint/classroom-python-for-finance-2025/refs/heads/main/Friends_Transcript.txt
file_path = "Friends_Transcript.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()
len_text = len(text)
unique_characters = sorted(list(set(text)))
vocabulary_size = len(unique_characters)
print(f"Length of {file_path}: {len_text:,} characters.")
print(text[:1000])
print(unique_characters)
print(vocabulary_size)
# Mapping string to integers
class EncoderAndDecoder:
    def __init__(self, unique_characters: list):
        self._str_to_int = {ch: i for i, ch in enumerate(unique_characters)}
        self._int_to_str = {i: ch for i, ch in enumerate(unique_characters)}
    def encode(self, string: str) -> list:
        return [self._str_to_int[char] for char in string]
    def decode(self, integers: list) -> str:
        strings = [self._int_to_str[integer] for integer in integers]
        return "".join(strings)

encoder_and_decoder = EncoderAndDecoder(unique_characters)
encoded = encoder_and_decoder.encode("Chandler: Sounds like a date to me.")
print(encoded)
print(len(encoded))
print(len("Chandler: Sounds like a date to me."))
print(encoder_and_decoder.decode(encoded))
# Mapping string to integers
import torch

class CustomDataLoader:
    # Train and validation data splits
    def __init__(self, encoded_text: list):
        self._encoded_text = encoded_text
    def train_valid_split(self):
        data = torch.tensor(self._encoded_text, dtype=torch.long)
        n = int(0.9 * len(data))
        train_data = data[:n] # First 90% will be train, rest valid
        valid_data = data[n:] # First 90% will be train, rest valid
        return train_data, valid_data
    # Generate a small batch of data of inputs x and targets y
    def get_batch(self, split: str, batch_size: int, block_size: int, device: str):
        train_data, valid_data = self.train_valid_split()
        if split == "train":
            split_data = train_data
        else:
            split_data = valid_data
        ix = torch.randint(len(split_data) - block_size, (batch_size,))
        x = torch.stack([split_data[i:(i + block_size)] for i in ix])
        y = torch.stack([split_data[(i + 1): (i + 1 + block_size)] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
# Attention: SingleHeadAttention
import torch.nn as nn
from torch.nn import functional as F

class SingleHeadAttention(nn.Module):
    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape # batch, time-step, channels
        k = self.key(x)
        q = self.query(x)
        # Compute attention weights(a way to represent similarities/correlations between embeddings)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # Weighted aggregation: Attention matrix(attention weights for all token embeddings)
        v = self.value(x)
        out = wei @ v
        return out
# Attention: MultiHeadAttention(Different dimensions to measure similarities/correlations between embeddings)
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
# Feed Forward layer
class FeedFoward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)
# A unit of Transformer
class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # Add & Norm(Resisual + LayerNorm)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
# Language Model
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocabulary_size)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx: int, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

batch_size = 64     # How many independent sequences will we process
block_size = 256    # The maximum context length for predictions
max_iters = 5000    # Training epochs
eval_interval = 500 # How many iters to print loss
learning_rate = 3e-4
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 4
dropout = 0.2

# Initialize model
model = LanguageModel()
m = model.to(device)
n_params_in_million = sum(p.numel() for p in m.parameters()) / 1e6
encoder_and_decoder = EncoderAndDecoder(unique_characters)
encoded_text = encoder_and_decoder.encode(text)
custom_data_loader = CustomDataLoader(encoded_text=encoded_text)
print(f"The language model has {n_params_in_million:,} million parameters.")

# Loss function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = custom_data_loader.get_batch(split, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}")
    # sample a batch of data
    xb, yb = custom_data_loader.get_batch(split="train", batch_size=batch_size, block_size=block_size, device=device)
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Predict/Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(encoder_and_decoder.decode(m.generate(context, max_new_tokens=500)[0].tolist()))