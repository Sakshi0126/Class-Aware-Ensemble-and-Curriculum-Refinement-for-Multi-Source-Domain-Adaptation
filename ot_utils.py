import ot
import numpy as np
from ot import emd2, sinkhorn2
from gensim.models import KeyedVectors

# === OT/Word2Vec utilities ===
def get_class_embeddings(class_names, w2v_model):
    embeddings = []
    for cls in class_names:
        if cls in w2v_model:
            embeddings.append(w2v_model[cls])
        else:
            embeddings.append(np.random.randn(w2v_model.vector_size))
    return np.stack(embeddings)

def build_cost_matrix(class_names, w2v_model):
    emb = get_class_embeddings(class_names, w2v_model)
    sim = 1 - np.dot(emb, emb.T) / (np.linalg.norm(emb, axis=1)[:,None] * np.linalg.norm(emb, axis=1)[None,:])
    return sim

def stable_softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x)
    s = np.sum(e_x, axis=-1, keepdims=True)
    return e_x / np.clip(s, 1e-8, None)

def wasserstein_distance_with_cost(pred1, pred2, cost_matrix, use_sinkhorn=False, reg=0.05, logger=None, step=None):
    pred1 = np.clip(pred1, 1e-8, 1.0)
    pred2 = np.clip(pred2, 1e-8, 1.0)
    pred1 = pred1 / pred1.sum()
    pred2 = pred2 / pred2.sum()
    if use_sinkhorn:
        if logger is not None:
            logger.info(f"Sinkhorn OT used at step {step} with reg={reg}")
        return sinkhorn2(pred1, pred2, cost_matrix, reg)
    else:
        return emd2(pred1, pred2, cost_matrix)

def compute_ot_weights(predictions_list, ensemble_avg, cost_matrix, confidence_list, diversity_list, alpha, beta, gamma, logger=None, step=None):
    weights = []
    fallback = False
    use_sinkhorn = getattr(compute_ot_weights, 'use_sinkhorn', False)
    sinkhorn_reg = getattr(compute_ot_weights, 'sinkhorn_reg', 0.05)

    for i, pred in enumerate(predictions_list):
        try:
            wdist = wasserstein_distance_with_cost(
                pred, ensemble_avg, cost_matrix,
                use_sinkhorn=use_sinkhorn,
                reg=sinkhorn_reg,
                logger=logger, step=step
            )
        except Exception as e:
            wdist = 1.0
            fallback = True
        score = -alpha * wdist + beta * confidence_list[i] + gamma * diversity_list[i]
        weights.append(score)
    weights = np.array(weights)
    if not np.all(np.isfinite(weights)) or np.allclose(weights, 0):
        fallback = True
        weights = np.ones_like(weights)
    weights = np.exp(weights - np.max(weights))
    s = np.sum(weights)
    if s < 1e-8 or not np.isfinite(s):
        fallback = True
        weights = weights / len(weights)
    else:
        
        weights = weights / s
        print("weights, wdist", weights, wdist)
        
    if fallback and logger is not None:
        logger.warning(f"OT fallback to uniform weights at step {step}")
    return weights
