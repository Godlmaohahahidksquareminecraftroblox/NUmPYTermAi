import numpy as np
import os, math, pickle, signal

# --- Hyperparameters ---
SAVE_PATH = "model_attn.npz"
ADAM_PATH = "adam_attn.pkl"

embed_dim = 192          # larger embedding
context_length = 64
mlp_hidden = 512         # scale up MLP with embedding
num_layers = 4           # 4 transformer blocks

learning_rate = 1e-4     # a bit smaller to prevent overfitting
weight_decay = 5e-4      # slightly higher regularization
beta1, beta2 = 0.9, 0.98 # slightly more stable Adam betas
eps = 1e-8

batch_size = 32          # larger batch can help smooth training
max_steps = 100000       # train longer since model is bigger
save_every = 1000
early_stop_patience = 20

grad_clip = 0.5          # clip gradients tighter for stability
dropout_prob = 0.5       # strong dropout to improve generalization
np.random.seed(42)

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
train_data = data[:int(0.9*n)]
val_data = data[int(0.9*n):]

def get_batch(src):
    ix = np.random.randint(0, len(src)-context_length-1, size=batch_size)
    x = np.stack([src[i:i+context_length] for i in ix])
    y = np.stack([src[i+1:i+context_length+1] for i in ix])
    return x, y

# --- Layer norm & dropout ---
def layer_norm(x, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    std = x.std(-1, keepdims=True)
    return (x-mean)/(std+eps), (x, mean, std, eps)

def layer_norm_backward(dout, cache):
    x, mean, std, eps = cache
    N = x.shape[-1]
    x_mu = x - mean
    inv_std = 1/(std+eps)
    dxhat = dout
    dvar = -0.5*(dxhat*x_mu).sum(-1, keepdims=True)*(inv_std**3)
    dmean = -dxhat.sum(-1, keepdims=True)*inv_std + dvar*(-2*x_mu).mean(-1, keepdims=True)
    dx = dxhat*inv_std + dvar*2*x_mu/N + dmean/N
    return dx

def dropout(x, p):
    mask = (np.random.rand(*x.shape) > p).astype(x.dtype)/(1-p)
    return x*mask, mask

# --- Transformer Block ---
class TransformerBlock:
    def __init__(self, D):
        self.wqkv = np.random.randn(D, D*3)/np.sqrt(D)
        self.wo = np.random.randn(D, D)/np.sqrt(D)
        self.w1 = np.random.randn(D, mlp_hidden)/np.sqrt(D)
        self.b1 = np.zeros(mlp_hidden)
        self.w2 = np.random.randn(mlp_hidden, D)/np.sqrt(mlp_hidden)
        self.b2 = np.zeros(D)

    def forward(self, x, train=True):
        B,T,D = x.shape
        qkv = x@self.wqkv
        q,k,v = np.split(qkv, 3, axis=-1)
        att_raw = q@k.transpose(0,2,1)/math.sqrt(D)
        mask = np.tril(np.ones((T,T)))
        att_raw = att_raw*mask + -1e9*(1-mask)
        att = np.exp(att_raw - att_raw.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        a = att@v@self.wo
        a,a_mask = dropout(a, dropout_prob) if train else (a, None)
        x1 = x+a
        x1, ln1_cache = layer_norm(x1)
        h = np.maximum(0, x1@self.w1 + self.b1)
        h,h_mask = dropout(h, dropout_prob) if train else (h, None)
        x2 = x1+h@self.w2 + self.b2
        x2, ln2_cache = layer_norm(x2)
        self.cache = (x, q, k, v, att, a, ln1_cache, h, ln2_cache, a_mask, h_mask, x1, h)
        return x2

    def backward(self, dout):
        x, q, k, v, att, a, ln1_cache, h, ln2_cache, a_mask, h_mask, x1, h_full = self.cache
        B,T,D = dout.shape

        dx2 = layer_norm_backward(dout, ln2_cache)
        dh2 = dx2@self.w2.T
        if h_mask is not None:
            dh2 *= h_mask
        dh2[h_full <= 0] = 0

        self.dw2 = (h_full.reshape(-1, mlp_hidden).T@dx2.reshape(-1,D))/(B*T)
        self.db2 = dx2.sum((0,1))/(B*T)

        dx1 = dh2@self.w1.T
        self.dw1 = (x1.reshape(-1,D).T@dh2.reshape(-1,mlp_hidden))/(B*T)
        self.db1 = dh2.sum((0,1))/(B*T)

        dx1 = dx1 + dx2@self.w2.T
        dx_res1 = layer_norm_backward(dx1, ln1_cache)
        da = dx_res1.copy()
        if a_mask is not None:
            da *= a_mask

        self.dwo = (att@v).reshape(-1,D).T@da.reshape(-1,D)/(B*T)
        dv = att.transpose(0,2,1)@da
        dat = da@v.transpose(0,2,1)
        dsoft = att*(dat-(dat*att).sum(-1, keepdims=True))
        dq = dsoft@k/np.sqrt(D)
        dk = dsoft.transpose(0,2,1)@q/np.sqrt(D)
        dqkv = np.concatenate([dq,dk,dv], axis=-1)
        self.dwqkv = (x.reshape(-1,D).T@dqkv.reshape(-1,D*3))/(B*T)
        dx = dqkv@self.wqkv.T + da@self.wo.T + dx_res1
        return dx
class TransformerBlock:
    def __init__(self, D):
        self.wqkv = np.random.randn(D, D * 3) / np.sqrt(D)
        self.wo = np.random.randn(D, D) / np.sqrt(D)
        self.w1 = np.random.randn(D, mlp_hidden) / np.sqrt(D)
        self.b1 = np.zeros(mlp_hidden)
        self.w2 = np.random.randn(mlp_hidden, D) / np.sqrt(mlp_hidden)
        self.b2 = np.zeros(D)

    def forward(self, x, train=True):
        B, T, D = x.shape
        qkv = x @ self.wqkv
        q, k, v = np.split(qkv, 3, axis=-1)
        att_raw = q @ k.transpose(0, 2, 1) / math.sqrt(D)
        mask = np.tril(np.ones((T, T)))
        att_raw = att_raw * mask + -1e9 * (1 - mask)
        att = np.exp(att_raw - att_raw.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        a = att @ v @ self.wo
        a, a_mask = dropout(a, dropout_prob) if train else (a, None)
        x1 = x + a
        x1, ln1_cache = layer_norm(x1)

        h_full = np.maximum(0, x1 @ self.w1 + self.b1)
        h, h_mask = dropout(h_full, dropout_prob) if train else (h_full, None)
        x2 = x1 + h @ self.w2 + self.b2
        x2, ln2_cache = layer_norm(x2)

        self.cache = (
            x, q, k, v, att, a, a_mask,
            x1, ln1_cache,
            h_full, h_mask,
            ln2_cache
        )
        return x2

    def backward(self, dout):
        (x, q, k, v, att, a, a_mask,
         x1, ln1_cache,
         h_full, h_mask,
         ln2_cache) = self.cache
        B, T, D = dout.shape

        # Backward through last layer norm
        dx2 = layer_norm_backward(dout, ln2_cache)

        # Backward through MLP output
        dh2 = dx2 @ self.w2.T
        if h_mask is not None:
            dh2 *= h_mask
        dh2[h_full <= 0] = 0  # ReLU backward

        self.dw2 = (h_full.reshape(-1, mlp_hidden).T @ dx2.reshape(-1, D)) / (B*T)
        self.db2 = dx2.sum((0, 1)) / (B*T)

        dx1 = dh2 @ self.w1.T
        self.dw1 = (x1.reshape(-1, D).T @ dh2.reshape(-1, mlp_hidden)) / (B*T)
        self.db1 = dh2.sum((0, 1)) / (B*T)

        # Backward through first layer norm
        dx1 = layer_norm_backward(dx1, ln1_cache)

        # Backward through attention
        da = dx1.copy()  # residual path
        if a_mask is not None:
            da *= a_mask

        self.dwo = (att @ v).reshape(-1, D).T @ da.reshape(-1, D) / (B*T)
        dv = att.transpose(0, 2, 1) @ da
        dat = da @ v.transpose(0, 2, 1)

        dsoft = att * (dat - (dat * att).sum(-1, keepdims=True))
        dq = dsoft @ k / math.sqrt(D)
        dk = dsoft.transpose(0, 2, 1) @ q / math.sqrt(D)

        dqkv = np.concatenate([dq, dk, dv], axis=-1)
        self.dwqkv = (x.reshape(-1, D).T @ dqkv.reshape(-1, D * 3)) / (B*T)

        dx = dqkv @ self.wqkv.T + da @ self.wo.T
        return dx
# --- Model ---
class MiniTransformer:
    def __init__(self):
        D = embed_dim
        self.token_embed = np.random.randn(vocab_size, D)/np.sqrt(D)
        self.pos_embed = np.random.randn(1, context_length, D)/np.sqrt(D)
        self.blocks = [TransformerBlock(D) for _ in range(num_layers)]
        self.head = np.random.randn(D, vocab_size)/np.sqrt(D)
        self.bh = np.zeros(vocab_size)

    def forward(self, idx, train=True):
        B,T = idx.shape
        self.last_idx = idx
        x = self.token_embed[idx] + self.pos_embed[:, :T, :]
        x,_ = dropout(x, dropout_prob) if train else (x, None)
        for blk in self.blocks:
            x = blk.forward(x, train)
        self.x = x
        return x@self.head + self.bh

    def backward(self, dlogits):
        B,T,D = self.x.shape
        self.dhead = self.x.reshape(-1,D).T@dlogits.reshape(-1,vocab_size)/(B*T)
        self.dbh = dlogits.sum((0,1))/(B*T)
        dx = dlogits@self.head.T
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)
        self.dtoken_embed = np.zeros_like(self.token_embed)
        for b in range(B):
            for t in range(T):
                self.dtoken_embed[self.last_idx[b,t]] += dx[b,t]
        self.dtoken_embed /= (B*T)
        self.dpos_embed = dx.sum(0, keepdims=True)/B

    def get_weights(self):
        w = {"token_embed": self.token_embed, "pos_embed": self.pos_embed,
             "head": self.head, "bh": self.bh}
        for i, blk in enumerate(self.blocks):
            w.update({f"blk{i}_wqkv": blk.wqkv, f"blk{i}_wo": blk.wo,
                      f"blk{i}_w1": blk.w1, f"blk{i}_b1": blk.b1,
                      f"blk{i}_w2": blk.w2, f"blk{i}_b2": blk.b2})
        return w

    def set_weights(self, d):
        self.token_embed = d["token_embed"]
        self.pos_embed = d["pos_embed"]
        self.head = d["head"]
        self.bh = d["bh"]
        self.blocks = [TransformerBlock(embed_dim) for _ in range(num_layers)]
        for i, blk in enumerate(self.blocks):
            blk.wqkv = d[f"blk{i}_wqkv"]; blk.wo = d[f"blk{i}_wo"]
            blk.w1 = d[f"blk{i}_w1"]; blk.b1 = d[f"blk{i}_b1"]
            blk.w2 = d[f"blk{i}_w2"]; blk.b2 = d[f"blk{i}_b2"]

# --- Loss & save/load ---
def cross_entropy_loss(logits, targets):
    B,T,V = logits.shape
    logits = logits - logits.max(-1, keepdims=True)
    probs = np.exp(logits); probs /= probs.sum(-1, keepdims=True)
    idx = (np.arange(B)[:,None], np.arange(T), targets)
    logp = -np.log(np.clip(probs[idx],1e-9,1.0))
    loss = logp.mean()
    dlogits = probs; dlogits[idx] -= 1; dlogits /= (B*T)
    return loss, dlogits

def save_npz(model, step, adam_state):
    np.savez(SAVE_PATH, step=step, **model.get_weights())
    with open(ADAM_PATH,"wb") as f: pickle.dump(adam_state,f)
    print(f"💾 Saved at step {step}")

def load_npz():
    if not os.path.exists(SAVE_PATH): return None,0,{}
    z = np.load(SAVE_PATH, allow_pickle=True)
    step = int(z["step"])
    w = {k:z[k] for k in z.files if k!="step"}
    model = MiniTransformer()
    model.set_weights(w)
    adam_state = pickle.load(open(ADAM_PATH,"rb")) if os.path.exists(ADAM_PATH) else {}
    print(f"📂 Loaded at step {step}")
    return model,step,adam_state

# --- Signal handling ---
def handle_interrupt(sig,frame):
    print("\n⛔ Interrupt — saving...")
    save_npz(model,current_step,adam_state); exit()
signal.signal(signal.SIGINT, handle_interrupt)

# --- Train ---
model,start_step,adam_state = load_npz()
if model is None: model = MiniTransformer()
for param,val in model.get_weights().items():
    if param not in adam_state: adam_state[param] = {"m":np.zeros_like(val),"v":np.zeros_like(val)}

ema = 0.0; best_vloss = float('inf'); patience = 0
for step in range(start_step, max_steps):
    current_step = step
    x,y = get_batch(train_data)
    logits = model.forward(x, train=True)
    loss,dlogits = cross_entropy_loss(logits,y)
    ema = loss if step==0 else 0.9*ema+0.1*loss
    if step%10==0:
        print(f"[{step}] loss={loss:.4f} ema={ema:.4f} ppl={math.exp(loss):.2f}")

    model.backward(dlogits)

    grads = {
        "token_embed":model.dtoken_embed, "pos_embed":model.dpos_embed,
        "head":model.dhead, "bh":model.dbh
    }
    for i,blk in enumerate(model.blocks):
        grads[f"blk{i}_wqkv"]=blk.dwqkv; grads[f"blk{i}_wo"]=blk.dwo
        grads[f"blk{i}_w1"]=blk.dw1; grads[f"blk{i}_b1"]=blk.db1
        grads[f"blk{i}_w2"]=blk.dw2; grads[f"blk{i}_b2"]=blk.db2

    # Adam update
    t = step+1
    for k,g in grads.items():
        if np.linalg.norm(g) > grad_clip:
            g *= grad_clip/np.linalg.norm(g)
        p = model.get_weights()[k]
        st = adam_state[k]
        st["m"] = beta1*st["m"]+(1-beta1)*g
        st["v"] = beta2*st["v"]+(1-beta2)*(g*g)
        mhat = st["m"]/(1-beta1**t); vhat = st["v"]/(1-beta2**t)
        new_p = p - learning_rate*(mhat/(np.sqrt(vhat)+eps) + weight_decay*p)
        if k.startswith("blk"):
            i = int(k[3]); name=k[5:]
            setattr(model.blocks[i], name, new_p)
        else:
            setattr(model, k, new_p)

    # Validation
    if step%save_every==0 and step>0:
        valx,valy = get_batch(val_data)
        vlogits = model.forward(valx, train=False)
        vloss,_ = cross_entropy_loss(vlogits,valy)
        print(f"⭐ val_loss={vloss:.4f} ppl={math.exp(vloss):.2f}")

        if vloss < best_vloss:
            best_vloss = vloss; patience = 0
            save_npz(model,step,adam_state)
        else:
            patience += 1
        if patience >= early_stop_patience:
            print("🛑 Early stopping")
            break

print("✅ Training complete!")
