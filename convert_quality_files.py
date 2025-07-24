import os
import pickle
from collections import defaultdict

import numpy as np

from config.config import config as cfg

authentic = cfg.auth_dict
synthetic = cfg.synt_dict
method = "grafiqs"

names = {
    "crfiqa": "r100_quality.txt",
    "grafiqs": "GraFIQs_block2.txt"
}

datasets = authentic | synthetic


for k,v in datasets.items():

    v = v.replace("/images", f"/{names[method]}")

    # Path to your txt file
    txt_file = v

    # Dictionary to store class-wise scores
    class_scores = defaultdict(list)

    # Read and process the file
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as f:
            for line in f:
                path, score = line.strip().split()
                score = float(score)
                class_id = path.split('/')[-2]  # Extract class (second last part of path)
                class_scores[class_id].append(score)

        # Compute mean per class
        class_mean_scores = {cls: sum(scores) / len(scores) for cls, scores in class_scores.items()}
        class_std_scores = {cls: np.std(scores) for cls, scores in class_scores.items()}

        class_mean_scores = (list(class_mean_scores.keys()), list(class_mean_scores.values()))
        class_std_scores = (list(class_std_scores.keys()), list(class_std_scores.values()))

        base_putpath = f"metrics/data2/{k}/"

        os.makedirs(base_putpath, exist_ok=True)
        with open(os.path.join(base_putpath, f"quality_{method}_mean.pkl"), 'wb') as fp:
            pickle.dump(class_mean_scores, fp)

        with open(os.path.join(base_putpath, f"quality_{method}_std.pkl"), 'wb') as fp2:
            pickle.dump(class_std_scores, fp2)
        # np.save(output_path, to_save)

        print(f"Results saved in: {base_putpath}")


