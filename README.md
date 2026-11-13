## 1. Architecture & Engineering Highlights

* **Bipartite Node-Offset Wiring:** Maps 943 users to node indices `[0, 942]` and shifts 1,682 items by `+943` to `[943, 2624]` within a unified 2,625-node graph. Interactions are stacked bidirectionally (User <-> Item) into an `edge_index` tensor of shape `[2, 181140]`.
* **Vectorized Rejection Negative Sampling:** Uses an O(1) boolean lookup matrix (`train_mat` of shape `[943, 1682]`) and iterative boolean masking (`while bad.any()`) to resample collided items exclusively for affected batch slots without Python `for` loops.
* **Bayesian Personalized Ranking (BPR) & Layer-0 L2 Regularization:** Optimizes pairwise ranking margins by penalizing instances where an unwatched item (`neg_score`) scores higher than a watched item (`pos_score`). Rather than using global optimizer `weight_decay` (which penalizes inactive item embeddings every step), L2 regularization is applied strictly to active batch indices in the initial learnable Layer-0 embedding table (`e0 = model.embedding.weight`):

$$
\mathcal{L}_{\text{total}} = -\frac{1}{|B|} \sum_{(u, \text{pos}, \text{neg}) \in B} \ln \sigma(\text{pos\_score} - \text{neg\_score}) + \lambda \frac{\|e_u^{(0)}\|_2^2 + \|e_{\text{pos}}^{(0)}\|_2^2 + \|e_{\text{neg}}^{(0)}\|_2^2}{2|B|}
$$

* **Loop-Free Top-K Ranking Evaluation:** Computes all-pairs preference scores via matrix multiplication (`emb[:n_users] @ emb[n_users:].T`), masks seen training interactions with `-inf`, extracts top-20 candidates via `.topk()`, and calculates **Recall@20** and **NDCG@20** using 2D tensor broadcasting (`torch.gather` and broadcasted `IDCG` masks).
* **Single-Batch Overfit Sanity Check:** Includes a `SANITY_OVERFIT` diagnostic mode that verifies gradient flow by driving BPR loss toward `0.0000` on a fixed mini-batch of 1,024 interactions prior to full-scale training.
