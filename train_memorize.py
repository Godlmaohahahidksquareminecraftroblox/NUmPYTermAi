import numpy as np
import os, math, pickle, signal

# --- Hyperparameters ---
# --- Improved Hyperparameters ---
SAVE_PATH = "model_attn.npz"
ADAM_PATH = "adam_attn.pkl"

embed_dim = 128              # Higher embedding size for better memorization
context_length = 32          # Longer context helps with remembering sequences
mlp_hidden = 512             # Wider MLP for greater capacity to memorize patterns

learning_rate = 2e-4         # Faster convergence without instability
batch_size = 8               # Larger batch for better gradient estimates (adjust if RAM-limited)
max_steps = 30000            # More steps for deeper memorization
save_every = 1000            # Less frequent saves = faster training (unless you expect interruptions)

grad_clip = 1.0              # Prevent exploding gradients
beta1, beta2 = 0.9, 0.99     # Slightly faster adaptation on recent gradients
eps = 1e-8                   # Adam epsilon for numerical stability
dropout_prob = 0.2           # Disable dropout for pure memorization

num_layers = 6               # More layers = deeper model with more memory capacity

np.random.seed(42)           # Reproducibility

# --- Data Loading ---
with open("dataset.txt", "r", encoding="utf-8") as f:
    data_text = f.read()
chars = sorted(set(data_text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
enc = lambda s: [stoi[c] for c in s]
dec = lambda arr: ''.join(itos[i] for i in arr)
data = np.array(enc(data_text), dtype=np.int32)
n = len(data)
train_data = data[:int(0.9 * n)]
val_data = data[int(0.9 * n):]

def get_batch(src):
    ix = np.random.randint(0, len(src) - context_length - 1, size=batch_size)
    x = np.stack([src[i:i+context_length] for i in ix])
    y = np.stack([src[i+1:i+context_length+1] for i in ix])
    return x, y

# --- Layer Norm ---
def layer_norm(x, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    std = x.std(-1, keepdims=True)
    norm = (x - mean) / (std + eps)
    cache = (x, mean, std, eps)
    return norm, cache

def layer_norm_backward(dout, cache):
    x, mean, std, eps = cache
    N = x.shape[-1]
    x_mu = x - mean
    inv_std = 1. / (std + eps)
    dxhat = dout
    dvar = (-0.5 * (dxhat * x_mu).sum(-1, keepdims=True)) * (inv_std**3)
    dmean = -dxhat.sum(-1, keepdims=True) * inv_std + dvar * (-2. * x_mu).mean(-1, keepdims=True)
    dx = dxhat * inv_std + dvar * 2 * x_mu / N + dmean / N
    return dx

def dropout(x, p):
    mask = (np.random.rand(*x.shape) > p).astype(x.dtype) / (1 - p)
    return x * mask, mask

# --- Transformer Block ---
class TransformerBlock:
    def __init__(self, D):
        self.wqkv = np.random.randn(D, D * 3).astype(np.float32) / np.sqrt(D)
        self.wo = np.random.randn(D, D).astype(np.float32) / np.sqrt(D)
        self.w1 = np.random.randn(D, mlp_hidden).astype(np.float32) / np.sqrt(D)
        self.b1 = np.zeros(mlp_hidden, dtype=np.float32)
        self.w2 = np.random.randn(mlp_hidden, D).astype(np.float32) / np.sqrt(mlp_hidden)
        self.b2 = np.zeros(D, dtype=np.float32)
        self.cache = {}

    def forward(self, x, train=True):
        B, T, D = x.shape
        qkv = x @ self.wqkv
        q, k, v = np.split(qkv, 3, axis=-1)
        att_raw = (q @ k.transpose(0, 2, 1)) / math.sqrt(D)
        mask = np.tril(np.ones((T, T), dtype=att_raw.dtype))
        att_raw = att_raw * mask + -1e9 * (1 - mask)
        att = np.exp(att_raw - att_raw.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        a = att @ v @ self.wo
        a, att_mask = dropout(a, dropout_prob) if train else (a, None)
        x_res1 = x + a
        x1, ln1_cache = layer_norm(x_res1)
        h = np.maximum(0, x1 @ self.w1 + self.b1)
        h, h_mask = dropout(h, dropout_prob) if train else (h, None)
        x_res2 = x1 + h @ self.w2 + self.b2
        x2, ln2_cache = layer_norm(x_res2)
        self.cache = dict(x=x, q=q, k=k, v=v, att=att, a=a, x1=x1, h=h, x2=x2,
                          ln1_cache=ln1_cache, ln2_cache=ln2_cache,
                          att_mask=att_mask, h_mask=h_mask)
        return x2

    def backward(self, dout):
        B, T, D = dout.shape
        c = self.cache
        dx2 = layer_norm_backward(dout, c["ln2_cache"])
        dh2 = dx2 @ self.w2.T
        if c["h_mask"] is not None:
            dh2 *= c["h_mask"]
        dh2[c["h"] <= 0] = 0
        self.dw2 = (c["h"].reshape(-1, mlp_hidden).T @ dx2.reshape(-1, D)) / (B * T)
        self.db2 = dx2.sum((0, 1)) / (B * T)
        self.dw1 = (c["x1"].reshape(-1, D).T @ dh2.reshape(-1, mlp_hidden)) / (B * T)
        self.db1 = dh2.sum((0, 1)) / (B * T)
        dx1 = dh2 @ self.w1.T + dx2
        dx_res1 = layer_norm_backward(dx1, c["ln1_cache"])
        da = dx_res1.copy()
        if c["att_mask"] is not None:
            da *= c["att_mask"]
        self.dwo = (c["att"] @ c["v"]).reshape(-1, D).T @ da.reshape(-1, D) / (B * T)
        dv = c["att"].transpose(0, 2, 1) @ da
        dat = da @ c["v"].transpose(0, 2, 1)
        dsoft = c["att"] * (dat - (dat * c["att"]).sum(-1, keepdims=True))
        dq = dsoft @ c["k"] / math.sqrt(D)
        dk = dsoft.transpose(0, 2, 1) @ c["q"] / math.sqrt(D)
        dqkv = np.concatenate([dq, dk, dv], axis=-1)
        self.dwqkv = c["x"].reshape(-1, D).T @ dqkv.reshape(-1, D * 3) / (B * T)
        dx = dqkv @ self.wqkv.T + da @ self.wo.T + dx_res1
        return dx

# --- Model ---
class MiniTransformer:
    def __init__(self):
        D = embed_dim
        self.token_embed = np.random.randn(vocab_size, D).astype(np.float32) / np.sqrt(D)
        self.pos_embed = np.random.randn(1, context_length, D).astype(np.float32) / np.sqrt(D)
        self.blocks = [TransformerBlock(D) for _ in range(num_layers)]
        self.head = np.random.randn(D, vocab_size).astype(np.float32) / np.sqrt(D)
        self.bh = np.zeros(vocab_size, dtype=np.float32)

    def forward(self, idx, train=True):
        B, T = idx.shape
        self.last_idx = idx
        x = self.token_embed[idx] + self.pos_embed[:, :T, :]
        for blk in self.blocks:
            x = blk.forward(x, train)
        self.x = x
        return x @ self.head + self.bh

    def backward(self, dlogits):
        B, T, D = self.x.shape
        self.dhead = self.x.reshape(-1, D).T @ dlogits.reshape(-1, vocab_size) / (B * T)
        self.dbh = dlogits.sum((0, 1)) / (B * T)
        dx = dlogits @ self.head.T
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)
        self.dtoken_embed = np.zeros_like(self.token_embed)
        for b in range(B):
            for t in range(T):
                self.dtoken_embed[self.last_idx[b, t]] += dx[b, t]
        self.dtoken_embed /= (B * T)
        self.dpos_embed = dx.sum(0, keepdims=True) / B

    def get_weights(self):
        out = {"token_embed": self.token_embed, "pos_embed": self.pos_embed,
               "head": self.head, "bh": self.bh}
        for i, blk in enumerate(self.blocks):
            out.update({
                f"blk{i}_wqkv": blk.wqkv, f"blk{i}_wo": blk.wo,
                f"blk{i}_w1": blk.w1, f"blk{i}_b1": blk.b1,
                f"blk{i}_w2": blk.w2, f"blk{i}_b2": blk.b2
            })
        return out

    def set_weights(self, d):
        self.token_embed = d["token_embed"]
        self.pos_embed = d["pos_embed"]
        self.head = d["head"]
        self.bh = d["bh"]
        self.blocks = [TransformerBlock(embed_dim) for _ in range(num_layers)]
        for i, blk in enumerate(self.blocks):
            blk.wqkv = d[f"blk{i}_wqkv"]
            blk.wo = d[f"blk{i}_wo"]
            blk.w1 = d[f"blk{i}_w1"]
            blk.b1 = d[f"blk{i}_b1"]
            blk.w2 = d[f"blk{i}_w2"]
            blk.b2 = d[f"blk{i}_b2"]

# --- Loss ---
def cross_entropy_loss(logits, targets):
    B, T, V = logits.shape
    logits = logits - logits.max(-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(-1, keepdims=True)
    idx = np.arange(B)[:, None], np.arange(T), targets
    logp = -np.log(np.clip(probs[idx], 1e-9, 1.0))
    loss = logp.mean()
    dlogits = probs
    dlogits[idx] -= 1
    dlogits /= (B * T)
    return loss, dlogits

# --- Checkpoint I/O ---
def save_npz(model, step, adam_state):
    np.savez(SAVE_PATH, step=step, **model.get_weights())
    with open(ADAM_PATH, "wb") as f:
        pickle.dump(adam_state, f)
    print(f"\U0001f4be Saved at step {step}")

def load_npz():
    if not os.path.exists(SAVE_PATH):
        return None, 0, {}
    z = np.load(SAVE_PATH, allow_pickle=True)
    step = int(z["step"])
    W = {k: z[k] for k in z.files if k != "step"}
    model = MiniTransformer()
    model.set_weights(W)
    adam_state = pickle.load(open(ADAM_PATH, "rb")) if os.path.exists(ADAM_PATH) else {}
    print(f"📂 Loaded from step {step}")
    return model, step, adam_state

# --- Signal handling ---
def handle_interrupt(sig, frame):
    print("\n⛔ Keyboard interrupt — saving checkpoint…")
    save_npz(model, current_step, adam_state)
    exit()
signal.signal(signal.SIGINT, handle_interrupt)

# --- Training Loop ---
model, start_step, adam_state = load_npz()
if model is None:
    model = MiniTransformer()
for param, val in model.get_weights().items():
    if param not in adam_state:
        adam_state[param] = {"m": np.zeros_like(val), "v": np.zeros_like(val)}

ema = 0.0
for step in range(start_step, max_steps):
    current_step = step
    x, y = get_batch(train_data)
    logits = model.forward(x, train=True)
    loss, dlogits = cross_entropy_loss(logits, y)
    ema = loss if step == 0 else 0.9 * ema + 0.1 * loss
    if step % 10 == 0:
        print(f"[{step:5d}] loss={loss:.4f} ema={ema:.4f} ppl={math.exp(loss):.2f}")
    model.backward(dlogits)

    t = step + 1
    grads = {
        **{k: getattr(model, "d" + k) for k in ["token_embed", "pos_embed", "head", "bh"]},
        **{f"blk{i}_{name}": getattr(blk, "d" + name) for i, blk in enumerate(model.blocks)
           for name in ["wqkv", "wo", "w1", "b1", "w2", "b2"]}
    }
    for k, g in grads.items():
        if np.linalg.norm(g) > grad_clip:
            g *= grad_clip / np.linalg.norm(g)
        st = adam_state[k]
        st["m"] = beta1 * st["m"] + (1 - beta1) * g
        st["v"] = beta2 * st["v"] + (1 - beta2) * (g * g)
        mhat = st["m"] / (1 - beta1 ** t)
        vhat = st["v"] / (1 - beta2 ** t)
        if k.startswith("blk"):
            i = int(k[3])
            name = k[5:]
            p = getattr(model.blocks[i], name)
            setattr(model.blocks[i], name, p - learning_rate * mhat / (np.sqrt(vhat) + eps))
        else:
            p = getattr(model, k)
            setattr(model, k, p - learning_rate * mhat / (np.sqrt(vhat) + eps))

    if step % save_every == 0 and step > 0:
        valx, valy = get_batch(val_data)
        v_logit = model.forward(valx, train=False)
        vloss, _ = cross_entropy_loss(v_logit, valy)
        print(f"⭐ Val loss: {vloss:.4f}, ppl={math.exp(vloss):.2f}")
        save_npz(model, step, adam_state)

print("✅ Training complete")
save_npz(model, max_steps, adam_state)
