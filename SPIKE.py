import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F # stateless functional style operations for building neural networks
import matplotlib
 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.nn import LightGCN
 
# ----------------------------------------------------------------------------
# Config -- change one thing at a time, and write every change in your README.
# ----------------------------------------------------------------------------
SEED = 0
EMB_DIM = 64
NUM_LAYERS = 6
LR = 1e-3
BATCH = 1024
EPOCHS = 100  # start small; raise once the loss is clearly falling
REG = 1e-4  # L2 on the layer-0 embeddings
K = 20  # Recall@K / NDCG@K
EVAL_EVERY = 10
SANITY_OVERFIT = False # set True first: should drive the loss on ONE batch toward 0
 
torch.manual_seed(SEED)

 
# ----------------------------------------------------------------------------
# 1. Data. Decide the interaction rule ONCE (here: every rating = an interaction).
#    If you instead threshold (e.g. rating >= 4), do it here and in CodeBook.
# ----------------------------------------------------------------------------
cols = ["user", "item", "rating", "ts"]
train_df = pd.read_csv("ml-100k/ua.base", sep="\t", names=cols)
test_df = pd.read_csv("ml-100k/ua.test", sep="\t", names=cols)
 
n_users, n_items = 943, 1682
train_u = torch.tensor(train_df["user"].values - 1, dtype=torch.long)  # 0-based
train_i = torch.tensor(train_df["item"].values - 1, dtype=torch.long)
test_u = torch.tensor(test_df["user"].values - 1, dtype=torch.long)
test_i = torch.tensor(test_df["item"].values - 1, dtype=torch.long)

# Item node id = item index + n_users (the 943 offset you already know about).
src = torch.cat([train_u, train_i + n_users])
dst = torch.cat([train_i + n_users, train_u])
edge_index = torch.stack([src, dst])  # bidirectional: message passing needs both ways
 
train_mat = torch.zeros(n_users, n_items, dtype=torch.bool)
train_mat[train_u, train_i] = True
test_mat = torch.zeros(n_users, n_items)
test_mat[test_u, test_i] = 1.0
  
print("train interactions:", len(train_u), "| test interactions:", len(test_u))
print("edge_index shape:", tuple(edge_index.shape))  # expect (2, 2 * len(train_u))
 
# ----------------------------------------------------------------------------
# 2. Model + optimizer. The optimizer is the piece that was missing.
# ----------------------------------------------------------------------------
model = LightGCN(num_nodes=n_users + n_items, embedding_dim=EMB_DIM, num_layers=NUM_LAYERS)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
 
 
# ----------------------------------------------------------------------------
# 3. Negative sampling: items the user has NOT interacted with in the train set.
# ----------------------------------------------------------------------------
def sample_negatives(users):
    neg = torch.randint(0, n_items, (len(users),))
    bad = train_mat[users, neg]
    while bad.any():  # resample the collisions until none are left
        neg[bad] = torch.randint(0, n_items, (int(bad.sum()),))
        bad = train_mat[users, neg]
    return neg
 
 
# ----------------------------------------------------------------------------
# 4. One training step on a batch of (user, positive item) pairs.
# ----------------------------------------------------------------------------
def train_step(u, pos):
    neg = sample_negatives(u)
 
    emb = model.get_embedding(edge_index)  # [n_users + n_items, EMB_DIM], layer-combined
    e_u = emb[u]
    e_pos = emb[pos + n_users]
    e_neg = emb[neg + n_users]
 
    pos_score = (e_u * e_pos).sum(dim=1)  # dot product
    neg_score = (e_u * e_neg).sum(dim=1)
    bpr = -F.logsigmoid(pos_score - neg_score).mean()  # the "referee"
 
    e0 = model.embedding.weight  # layer-0 (initial) embeddings
    reg = (
        e0[u].pow(2).sum() + e0[pos + n_users].pow(2).sum() + e0[neg + n_users].pow(2).sum()
    ) / (2 * len(u))
    loss = bpr + REG * reg
 
    optimizer.zero_grad()  # forgetting zero_grad / backward / step is the classic "loss never moves" bug
    loss.backward()
    optimizer.step()
    return loss.item(), bpr.item()
 
 
# ----------------------------------------------------------------------------
# 5. Evaluation: rank every item per user, hide train items, compare with test items.
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(k=K):
    model.eval()
    emb = model.get_embedding(edge_index)
    scores = emb[:n_users] @ emb[n_users:].T  # [n_users, n_items]
    scores[train_mat] = -float("inf")  # never recommend what the user already has in train
    topk = scores.topk(k, dim=1).indices  # [n_users, k]
 
    hits = test_mat.gather(1, topk)  # 1.0 where the recommended item is in the user's test set
    n_test = test_mat.sum(dim=1)
    valid = n_test > 0
 
    recall = (hits.sum(dim=1) / n_test.clamp(min=1))[valid].mean()
 
    discount = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float))  # rank 1..k
    dcg = (hits * discount).sum(dim=1)
    ideal = (torch.arange(k)[None, :] < n_test[:, None]).float()  # best case: hits fill the top slots
    idcg = (ideal * discount).sum(dim=1)
    ndcg = (dcg / idcg.clamp(min=1e-9))[valid].mean()
 
    model.train()
    return recall.item(), ndcg.item()
 
 
# ----------------------------------------------------------------------------
# 6. Sanity check: can the model overfit ONE batch? If not, the bug is in the loop.
# ----------------------------------------------------------------------------
if SANITY_OVERFIT:
    idx = torch.arange(BATCH)
    u, pos = train_u[idx], train_i[idx]
    for step in range(300):
        loss, _ = train_step(u, pos)
        if step % 50 == 0:
            print(f"overfit step {step:3d} | loss {loss:.4f}")
    raise SystemExit("Sanity run finished. Set SANITY_OVERFIT = False for the real run.")
 
# ----------------------------------------------------------------------------
# 7. The real loop. Log raw batch loss AND the epoch mean; keep the noise visible.
# ----------------------------------------------------------------------------
batch_losses, epoch_losses, evals = [], [], []
 
for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(len(train_u))
    running = []
    for start in range(0, len(perm), BATCH):
        idx = perm[start : start + BATCH]
        loss, _ = train_step(train_u[idx], train_i[idx])
        running.append(loss)
    batch_losses.extend(running)
    epoch_losses.append(float(np.mean(running)))
 
    msg = f"epoch {epoch:3d} | loss {epoch_losses[-1]:.4f}"
    if epoch == 1 or epoch % EVAL_EVERY == 0:
        r, n = evaluate()
        evals.append((epoch, r, n))
        msg += f" | Recall@{K} {r:.4f} | NDCG@{K} {n:.4f}"
    print(msg)
 
# ----------------------------------------------------------------------------
# 8. Save the artefacts that go in the README: loss curve + metric table.
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(batch_losses, alpha=0.35, label="raw batch loss")
steps_per_epoch = int(np.ceil(len(train_u) / BATCH))
ax[0].plot(
    [(i + 0.5) * steps_per_epoch for i in range(len(epoch_losses))],
    epoch_losses,
    label="epoch mean",
)
ax[0].set_xlabel("training step")
ax[0].set_ylabel("BPR loss")
ax[0].set_title("Training loss")
ax[0].legend()
ax[1].plot([e[0] for e in evals], [e[1] for e in evals], marker="o", label=f"Recall@{K}")
ax[1].plot([e[0] for e in evals], [e[2] for e in evals], marker="o", label=f"NDCG@{K}")
ax[1].set_xlabel("epoch")
ax[1].set_title("Held-out ranking metrics")
ax[1].legend()
plt.tight_layout()
plt.savefig("loss_curve_6layer.png", dpi=150)
pd.DataFrame(evals, columns=["epoch", f"recall@{K}", f"ndcg@{K}"]).to_csv("metrics_6layer.csv", index=False)
print("saved loss_curve.png and metrics.csv")
