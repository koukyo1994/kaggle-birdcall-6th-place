import json
import re

import numpy as np
import pandas as pd

from pathlib import Path

from fastprogress import progress_bar

from src.dataset import NAME2CODE, BIRD_CODE, SCINAME2CODE


def create_ground_truth(train: pd.DataFrame):
    labels = np.zeros((len(train), 264), dtype=int)
    for i, row in progress_bar(train.iterrows(), total=len(train)):
        ebird_code = BIRD_CODE[row.ebird_code]
        labels[i, ebird_code] = 1

        secondary_labels = eval(row.secondary_labels)
        for sl in secondary_labels:
            if NAME2CODE.get(sl) is not None:
                second_code = NAME2CODE[sl]
                labels[i, BIRD_CODE[second_code]] = 1

        background = row["background"]
        if isinstance(background, str):
            academic_names = re.findall("\((.*>)\)", background)
            academic_names = list(
                filter(
                    lambda x: x is not None,
                    map(
                        lambda x: SCINAME2CODE.get(x),
                        academic_names
                    )
                )
            )
            for bl in academic_names:
                labels[i, BIRD_CODE[bl]] = 1
    columns = list(BIRD_CODE.keys())
    index = train["filename"].map(lambda x: x.replace(".mp3", ".wav")).values
    labels_df = pd.DataFrame(labels, index=index, columns=columns)
    return labels_df


if __name__ == "__main__":
    DATA_DIR = Path("input/birdsong-recognition")
    THRESHOLD = 0.9

    train = pd.read_csv(DATA_DIR / "train.csv")
    annotation = pd.read_csv("output/sed/000_Stage1/PANNsAtt_sed.csv")

    clipwise_labels = annotation.groupby("filename").max()
    gt_labels = create_ground_truth(train)

    indices = set(clipwise_labels.index.values.tolist())
    gt_labels_use = gt_labels[gt_labels.index.isin(indices)]

    pred_for_one_label_sample = clipwise_labels[gt_labels_use.sum(axis=1) == 1].sort_index()
    gt_for_one_label_sample = gt_labels_use[gt_labels_use.sum(axis=1) == 1].sort_index()

    found_label = {}
    for i, (filename, sample) in progress_bar(enumerate(pred_for_one_label_sample.iterrows()),
                                              total=len(pred_for_one_label_sample)):
        gt = gt_for_one_label_sample.loc[filename, :]
        found = set(sample[sample > THRESHOLD].index.values)
        gt_label = set(gt[gt > THRESHOLD].index.values)
        if found - gt_label:
            found_label[filename] = list(found - gt_label)

    with open(DATA_DIR / "additional_labels.json", "w") as f:
        json.dump(found_label, f)
