# test/data_loader.py
import os
import glob
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class DataLoaderHelper:
    def __init__(self, dataset_config, modalities=('fc1', 'fc2', 'sc1', 'sc2')):
        self.dataset_config = dataset_config
        self.modalities = modalities

    def _load_sample_file(self, path):
        # 支持 torch .pt 或 numpy .npz
        if path.endswith('.pt'):
            try:
                d = torch.load(path, map_location='cpu')
            except Exception:
                d = None
        else:
            d = None
        if d is None:
            try:
                import numpy as _np
                arr = _np.load(path, allow_pickle=True)
                d = dict(arr)
            except Exception:
                return None
        return d

    def _gather_from_folder(self, folder, label_value):
        samples = []
        files = sorted([os.path.join(folder,f) for f in os.listdir(folder) if f.endswith('.pt') or f.endswith('.npz')])
        for f in files:
            d = self._load_sample_file(f)
            if d is None:
                continue
            # try find fc1-like keys
            sample = {}
            # accept multiple naming conventions
            for k in ('fc1_x','fc1','fc1_mat'):
                if k in d:
                    sample['fc1'] = d[k]
                    break
            for k in ('fc2_x','fc2','fc2_mat'):
                if k in d:
                    sample['fc2'] = d[k]
                    break
            for k in ('sc1_x','sc1','sc1_mat'):
                if k in d:
                    sample['sc1'] = d[k]
                    break
            for k in ('sc2_x','sc2','sc2_mat'):
                if k in d:
                    sample['sc2'] = d[k]
                    break
            # label
            if 'y' in d:
                try:
                    lab = int(d['y'].item()) if hasattr(d['y'],'item') else int(d['y'])
                except Exception:
                    lab = int(label_value)
            else:
                lab = int(label_value)
            sample['y'] = lab
            sample['path'] = f
            if all(k in sample for k in ('fc1','fc2','sc1','sc2')):
                samples.append(sample)
            else:
                logger.debug(f"skip incomplete: {f}")
        return samples

    def load_all_data(self):
        all_samples = []
        for label_name, subdict in self.dataset_config.items():
            # label mapping: user expected NC=0, MCI=1 in original script
            label_value = 0 if label_name.upper().startswith('NC') or label_name.upper().startswith('CU') else 1
            # each subdict contains directories for modalities
            # assume directories exist
            # We will iterate files in fc1 directory and match by filename across modalities where possible
            fc1_dir = subdict.get('fc1')
            # gather samples from fc1 dir
            if fc1_dir is None or not os.path.isdir(fc1_dir):
                logger.warning(f"missing dir for {label_name} fc1: {fc1_dir}")
                continue
            samples = self._gather_from_folder(fc1_dir, label_value)
            # For each sample found, try to locate corresponding files in other modality dirs by basename
            for s in samples:
                base = os.path.basename(s['path']).split('.')[0]
                # try other modalities
                for mod in ('fc2','sc1','sc2'):
                    mod_dir = subdict.get(mod)
                    if mod_dir and os.path.isdir(mod_dir):
                        # try find file starting with base
                        cand = None
                        for ext in ('.pt','.npz'):
                            pth = os.path.join(mod_dir, base + ext)
                            if os.path.exists(pth):
                                cand = pth; break
                        if cand:
                            d = self._load_sample_file(cand)
                            # pick first matching key
                            for kcandidate, field in [('fc2_x','fc2'),('fc2','fc2'),('sc1_x','sc1'),('sc1','sc1'),('sc2_x','sc2'),('sc2','sc2')]:
                                if kcandidate in d and mod not in s:
                                    s[mod] = d[kcandidate]
                                    break
                all_samples.append(s)

        # now coerce to arrays
        if not all_samples:
            raise RuntimeError("no samples found by DataLoaderHelper")

        fc1_list, fc2_list, sc1_list, sc2_list, labels = [], [], [], [], []
        for s in all_samples:
            # convert tensors to numpy
            def to_np(x):
                if isinstance(x, torch.Tensor):
                    return x.cpu().numpy()
                return x
            fc1_list.append(to_np(s['fc1']).ravel())
            fc2_list.append(to_np(s['fc2']).ravel())
            sc1_list.append(to_np(s['sc1']).ravel())
            sc2_list.append(to_np(s['sc2']).ravel())
            labels.append(int(s['y']))

        fc1_arr = np.vstack(fc1_list)
        fc2_arr = np.vstack(fc2_list)
        sc1_arr = np.vstack(sc1_list)
        sc2_arr = np.vstack(sc2_list)
        labels = np.array(labels, dtype=int)

        logger.info(f"Loaded data: {len(labels)} samples")
        return fc1_arr, fc2_arr, sc1_arr, sc2_arr, labels