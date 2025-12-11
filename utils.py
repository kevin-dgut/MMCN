# test/utils.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import ttest_ind
import logging

logger = logging.getLogger(__name__)


def augment_fc_data(fc_matrix, augmentation_strength=0.05):
    noise = np.random.normal(0, augmentation_strength, fc_matrix.shape)
    np.fill_diagonal(noise, 0)
    augmented_matrix = fc_matrix + noise
    return np.clip(augmented_matrix, -1, 1)


def spatial_dimensionality_reduction(matrix, target_size=64):
    # matrix: numpy array (N,N) or torch tensor
    is_tensor = isinstance(matrix, torch.Tensor)
    if is_tensor:
        matrix = matrix.cpu().numpy()
    n = matrix.shape[0]
    if n <= target_size:
        return torch.from_numpy(matrix).float() if is_tensor else matrix
    indices = np.linspace(0, n - 1, target_size, dtype=int)
    reduced = matrix[indices][:, indices]
    if is_tensor:
        return torch.from_numpy(reduced).float()
    return reduced


def roi_node_strength_from_flattened_matrix(flat_matrices):
    n_samples, f = flat_matrices.shape
    N = int(round(np.sqrt(f)))
    if N * N != f:
        raise ValueError(f"无法将特征长度 {f} 重塑为方阵 (N*N)。")
    reshaped = flat_matrices.reshape(n_samples, N, N)
    strengths = np.mean(np.abs(reshaped), axis=2)
    return strengths


def compute_top10_rois_combined(fc1, fc2, sc1, sc2, labels, outdir, prefix='combined'):
    os.makedirs(outdir, exist_ok=True)
    s1 = roi_node_strength_from_flattened_matrix(fc1)
    s2 = roi_node_strength_from_flattened_matrix(fc2)
    s3 = roi_node_strength_from_flattened_matrix(sc1)
    s4 = roi_node_strength_from_flattened_matrix(sc2)
    combined = (s1 + s2 + s3 + s4) / 4.0
    labels = np.array(labels)
    idx_nc = np.where(labels == 0)[0]
    idx_mci = np.where(labels == 1)[0]
    if len(idx_nc) == 0 or len(idx_mci) == 0:
        logger.warning("无法计算显著性：NC 或 MCI 样本数为0")
        return None
    N = combined.shape[1]
    results = []
    for i in range(N):
        grp_nc = combined[idx_nc, i]
        grp_mci = combined[idx_mci, i]
        try:
            tstat, pval = ttest_ind(grp_nc, grp_mci, equal_var=False, nan_policy='omit')
        except Exception:
            tstat, pval = np.nan, np.nan
        results.append((i + 1, float(np.nanmean(grp_nc)), float(np.nanmean(grp_mci)), float(tstat), float(pval)))
    res_df = pd.DataFrame(results, columns=['roi', 'mean_nc', 'mean_mci', 't_stat', 'p_value'])
    res_df['mean_diff'] = res_df['mean_nc'] - res_df['mean_mci']
    nc_df = res_df[res_df['mean_diff'] > 0].sort_values(by='t_stat', ascending=False).reset_index(drop=True)
    mci_df = res_df[res_df['mean_diff'] < 0].sort_values(by='t_stat').reset_index(drop=True)
    out_csv = os.path.join(outdir, f'{prefix}_roi_significance_all.csv')
    res_df.to_csv(out_csv, index=False)
    out_nc = os.path.join(outdir, f'{prefix}_top10_NC.csv')
    out_mci = os.path.join(outdir, f'{prefix}_top10_MCI.csv')
    nc_df.head(10).to_csv(out_nc, index=False)
    mci_df.head(10).to_csv(out_mci, index=False)
    logger.info(f"ROI 显著性结果已保存: {out_csv}")
    return {'all': out_csv, 'nc': out_nc, 'mci': out_mci}


def save_heatmap(mat, outpath, title=None, cmap='hot'):
    plt.figure(figsize=(6,6))
    plt.imshow(mat, cmap=cmap)
    if title:
        plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()