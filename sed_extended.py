import json
import math

import numpy as np
import torch

import src.configuration as C
import src.dataset as dataset
import src.model as models
import src.utils as utils

from pathlib import Path
from fastprogress import progress_bar


if __name__ == "__main__":
    args = utils.get_parser().parse_args()
    config = utils.load_config(args.config)

    global_params = config["globals"]

    output_dir = Path(global_params["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    utils.set_seed(global_params["seed"])

    device = C.get_device(global_params["device"])

    _, datadir = C.get_metadata(config)
    df = C.get_additional_metadata(config)

    loader = C.get_sed_inference_loader(df, datadir, config)
    models_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(len(global_params["weights"])):
        models_dict[i] = models.get_model_for_inference(config,
                                                        global_params["weights"][i])
        models_dict[i].to(device)
        models_dict[i].eval()

    additional_labels_extended = {}
    for batch in progress_bar(loader):
        soft_labels = []
        if "waveform" in batch.keys():
            tensor = batch["waveform"]
        else:
            tensor = batch["image"]
        wav_name = batch["wav_name"][0]
        target = batch["targets"].detach().cpu().numpy()[0]
        duration = batch["duration"].detach().cpu().numpy()[0]
        period = batch["period"].detach().cpu().numpy()[0]
        global_time = 0.0
        if tensor.ndim == 3 and "waveform" in batch.keys():
            tensor = tensor.squeeze(0)
        elif tensor.ndim == 5 and "image" in batch.keys():
            tensor = tensor.squeeze(0)
        batch_size = 32
        whole_size = batch.size(0)

        if whole_size % batch_size == 0:
            n_iter = whole_size // batch_size
        else:
            n_iter = whole_size // batch_size + 1

        for index in range(n_iter):
            iter_batch = batch[index * batch_size:(index + 1) * batch_size]
            if iter_batch.ndim == 1:
                iter_batch = iter_batch.unsqueeze(0)
            elif iter_batch.ndim == 3:
                iter_batch = iter_batch.unsqueeze(0)
            iter_batch = iter_batch.to(device)

            predictions = []
            with torch.no_grad():
                for model in models_dict.values():
                    prediction = model(iter_batch)
                    segmentwise_output = prediction["segmentwise_output"].detach(
                        ).cpu().numpy()
                    predictions.append(segmentwise_output)

            segmentwise_outputs = np.mean(predictions, axis=0)
            for short_clip in segmentwise_outputs:
                if duration - global_time < period:
                    remain_seconds = duration - global_time
                    sec_per_segment = period / len(short_clip)
                    remain_index = math.ceil(remain_seconds / sec_per_segment)
                    short_clip = short_clip[:remain_index]
                if len(short_clip) > 0:
                    soft_labels.append(short_clip.astype(np.float16))
                global_time += period

        concatenated_soft_labels = np.concatenate(soft_labels, axis=0)
        filepath = output_dir / (wav_name + ".npy")
        np.save(filepath, concatenated_soft_labels)

        if target.sum() == 1:
            clipwise_label = concatenated_soft_labels.max(axis=0)
            gt_label = set(np.argwhere(target)[0].tolist())
            found = set(np.argwhere(clipwise_label > 0.9).reshape(-1))
            if found - gt_label:
                found_ = list(found - gt_label)
                found_ = [dataset.INV_BIRD_CODE[i] for i in found_]
                additional_labels_extended[wav_name] = found_

    with open(output_dir.parent / "additional_labels_extended.json", "w") as f:
        json.dump(additional_labels_extended, f)
