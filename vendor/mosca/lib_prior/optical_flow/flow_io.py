import os
import os.path as osp

import numpy as np


def save_flow_bundle(dst_dir, flow_records, frame_names=None, save_pair_files=True):
    os.makedirs(dst_dir, exist_ok=True)
    if not save_pair_files:
        return

    for record in flow_records:
        src_name = record["src_name"]
        dst_name = record["dst_name"]
        flow = record["flow"]
        mask = record["mask"]
        np.savez_compressed(
            osp.join(dst_dir, f"{src_name}_to_{dst_name}.npz"),
            flow=flow.astype(np.float16),
            mask=mask.astype(np.float16),
        )
