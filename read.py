import os
import numpy as np

base_path = "data/dvlog-dataset/dvlog-dataset"
ids = [292, 360,542,799, 811]

for _id in ids:
    _id = str(_id)
    v_path = os.path.join(base_path, _id, f"{_id}_visual.npy")
    v = np.load(v_path)
    print(_id, v.shape)
    if v.ndim == 1:
        print(v[:8])
    else:
        print(v[:8, :])