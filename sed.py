import numpy as np
import pandas as pd
import torch

import src.configuration as C
import src.dataset as dataset
import src.models as models
import src.utils as utils

from pathlib import Path

from fastprogress import progress_bar


if __name__ == "__main__":
    args = utils.get_sed_parser().parse_args()
    config = utils.load_config(args.config)

    global_params = config["globals"]

    output_dir = Path(global_params["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    utils.set_seed(global_params["seed"])
    device = C.get_device(global_params["device"])

    df, datadir = C.get_metadata(config)
    splitter = C.get_split(config)

    for i, (_, val_idx) in enumerate(splitter.split(df, y=df["ebird_code"])):
        if i not in global_params["folds"]:
            continue

        val_df = df.loc[val_idx, :].reset_index(drop=True)
        loader = C.get_sed_inference_loader(val_df, datadir, config)
        model = models.get_model_for_inference(config,
                                               global_params["weights"][i])

        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
            model.to(device)
        model.eval()

        estimated_event_list = []
        for batch in progress_bar(loader):
            waveform = batch["waveform"]
            ebird_code = batch["ebird_code"][0]
            wav_name = batch["wav_name"][0]
            target = batch["targets"].detach().cpu().numpy()[0]
            global_time = 0.0
            if waveform.ndim == 3:
                waveform = waveform.squeeze(0)

            batch_size = 32
            whole_size = waveform.size(0)
            if whole_size % batch_size == 0:
                n_iter = whole_size // batch_size
            else:
                n_iter = whole_size // batch_size + 1

            for index in range(n_iter):
                iter_batch = waveform[index * batch_size:(index + 1) * batch_size]
                if iter_batch.ndim == 1:
                    iter_batch = iter_batch.unsqueeze(0)
                iter_batch = iter_batch.to(device)
                with torch.no_grad():
                    prediction = model(iter_batch)
                    framewise_output = prediction["framewise_output"].detach(
                    ).cpu().numpy()

                thresholded = framewise_output >= args.threshold
                target_indices = np.argwhere(target).reshape(-1)
                for short_clip in thresholded:
                    for target_idx in target_indices:
                        if short_clip[:, target_idx].mean() == 0:
                            pass
                        else:
                            detected = np.argwhere(
                                short_clip[:, target_idx]).reshape(-1)
                            head_idx = 0
                            tail_idx = 0
                            while True:
                                if (tail_idx + 1 == len(detected)) or (
                                        detected[tail_idx + 1] -
                                        detected[tail_idx] != 1):
                                    onset = 0.01 * detected[head_idx] + global_time
                                    offset = 0.01 * detected[tail_idx] + global_time
                                    estimated_event = {
                                        "filename": wav_name,
                                        "ebird_code": dataset.INV_BIRD_CODE[target_idx],
                                        "onset": onset,
                                        "offset": offset
                                    }
                                    estimated_event_list.append(estimated_event)
                                    head_idx = tail_idx + 1
                                    tail_idx = tail_idx + 1
                                    if head_idx > len(detected):
                                        break
                                else:
                                    tail_idx = tail_idx + 1
                    global_time += 5.0

        estimated_event_df = pd.DataFrame(estimated_event_list)
        save_filename = global_params["save_path"].replace(".csv", "")
        save_filename += f"_th{args.threshold}" + ".csv"
        save_path = output_dir / save_filename
        if save_path.exists():
            event_level_labels = pd.read_csv(save_path)
            estimated_event_df = pd.concat(
                [event_level_labels, estimated_event_df], axis=0,
                sort=False).reset_index(drop=True)
            estimated_event_df.to_csv(save_path, index=False)
        else:
            estimated_event_df.to_csv(save_path, index=False)
