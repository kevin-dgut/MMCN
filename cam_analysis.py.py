# test/cam_analysis.py
import os
import json
import numpy as np
import torch
import glob
import logging
from test.data_loader import DataLoaderHelper
from test.model import SCG_ViT
from test.utils import save_heatmap
from test.gradcam_utils import compute_patch_level_cam, _project_cam_to_weights  

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def auto_select_checkpoint(root='/root/Project'):
    cands = glob.glob(os.path.join(root, '**', 'checkpoint*.pt'), recursive=True) + \
            glob.glob(os.path.join(root, '**', 'model*.pt'), recursive=True) + \
            glob.glob(os.path.join(root, '**', '*.pth'), recursive=True)
    matches = []
    for p in sorted(set(cands)):
        try:
            ck = torch.load(p, map_location='cpu')
            if isinstance(ck, dict):
                keys = list(ck.keys())
                if any('patch_embeddings' in k for k in keys):
                    matches.append(p)
        except Exception:
            continue
    if not matches:
        return None
    matches_sorted = sorted(matches, key=lambda p: os.path.getmtime(p), reverse=True)
    return matches_sorted[0]


def run_class_level_aggregation(dataset_config, model_ctor_kwargs, outdir='/root/Project/model/four_modalities', data_root='/root/Project/data/ADNI'):
    os.makedirs(outdir, exist_ok=True)
    vis_dir = os.path.join(outdir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    ckpt = auto_select_checkpoint()
    model = SCG_ViT(**model_ctor_kwargs)
    if ckpt:
        try:
            ck = torch.load(ckpt, map_location='cpu')
            state = None
            for k in ('model','state_dict','model_state_dict'):
                if k in ck and isinstance(ck[k], dict):
                    state = ck[k]; break
            if state is None:
                state = ck
            model.load_state_dict(state, strict=False)
            logger.info(f"Loaded checkpoint {ckpt} (strict=False)")
        except Exception as e:
            logger.warning("load checkpoint failed: " + str(e))
    else:
        logger.info("no checkpoint found, using random init model")

    # get A,b if available
    try:
        A = model.patch_embeddings.fc1_proj.weight.detach().cpu().numpy()
        b = model.patch_embeddings.fc1_proj.bias.detach().cpu().numpy() if model.patch_embeddings.fc1_proj.bias is not None else np.zeros(A.shape[0])
    except Exception:
        A, b = None, None

    loader = DataLoaderHelper(dataset_config)
    fc1, fc2, sc1, sc2, labels = loader.load_all_data()

    accum = {}
    counts = {}
    n_samples = fc1.shape[0]
    for i in range(n_samples):
        # build single-sample flattened arrays
        s_fc1 = fc1[i]
        s_fc2 = fc2[i]
        s_sc1 = sc1[i]
        s_sc2 = sc2[i]
        lab = int(labels[i])
        # compute patch-level cams using gradcam_utils (must exist)
        try:
            # prepare tensors as model expects (1,N) etc.
            inp = (torch.tensor(s_fc1).unsqueeze(0).float(),
                   torch.tensor(s_fc2).unsqueeze(0).float(),
                   torch.tensor(s_sc1).unsqueeze(0).float(),
                   torch.tensor(s_sc2).unsqueeze(0).float())
            cams, cam_embs, pred, probs = compute_patch_level_cam(model, inp, device='cpu')
        except Exception as e:
            logger.debug(f"skip sample {i} compute_patch_level_cam failed: {e}")
            continue
        cam_list = [v for v in cam_embs.values() if v is not None]
        if not cam_list:
            continue
        cam_vec = np.mean(np.stack(cam_list, axis=0), axis=0)
        if A is None:
            continue
        x = _project_cam_to_weights(A, b, cam_vec)
        # reconstruct W (prefer model num_rois)
        model_n = getattr(model, 'num_rois', getattr(model.patch_embeddings, 'num_rois', 116))
        n = int(model_n)
        W = None
        L = len(x)
        if L == n:
            W = np.outer(x, x); np.fill_diagonal(W, 0.0)
        elif L == n * n:
            W = np.asarray(x).reshape((n, n)); W = (W + W.T) / 2.0; np.fill_diagonal(W, 0.0)
        else:
            tri_len = n*(n-1)//2
            if L == tri_len:
                M = np.zeros((n,n)); iu = np.triu_indices(n, k=1); M[iu] = x; W = M + M.T
            else:
                try:
                    W = np.asarray(x).reshape((n,n)); W = (W + W.T) / 2.0; np.fill_diagonal(W, 0.0)
                except Exception:
                    continue
        absW = np.abs(W)
        accum.setdefault(lab, np.zeros_like(absW))
        counts.setdefault(lab, 0)
        accum[lab] += absW
        counts[lab] += 1

    # after loop compute top lists and save
    result = {}
    for lab, mat in accum.items():
        col_strength = mat.sum(axis=0)
        top_rois = list(np.argsort(col_strength)[-10:][::-1])
        tri = np.triu_indices(mat.shape[0], k=1)
        edge_vals = np.abs(mat[tri])
        top_edge_pos = np.argsort(edge_vals)[-20:][::-1]
        top_edges = [[int(tri[0][int(idx)]), int(tri[1][int(idx)]), float(mat[tri[0][int(idx)], tri[1][int(idx)]])] for idx in top_edge_pos]
        # save heatmap
        save_heatmap(mat, os.path.join(vis_dir, f'class_{lab}_accum_absW.png'), title=f'class_{lab}_accum_absW')
        result[lab] = {'num_samples': int(counts.get(lab,0)), 'top_rois': top_rois, 'top_edges': top_edges}

    out_summary = os.path.join(outdir, 'gradcam_class_level_summary.json')
    with open(out_summary, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote class-level summary to %s", out_summary)
    return result