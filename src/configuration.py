import json

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import sklearn.model_selection as sms

import src.dataset as datasets

from pathlib import Path

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from src.criterion import ImprovedPANNsLoss, ImprovedFocalLoss  # noqa
from src.transforms import (get_transforms, get_waveform_transforms,
                            get_spectrogram_transforms)


def get_device(device: str):
    return torch.device(device)


def get_optimizer(model: nn.Module, config: dict):
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config.get("name")

    return optim.__getattribute__(optimizer_name)(model.parameters(),
                                                  **optimizer_config["params"])


def get_scheduler(optimizer, config: dict):
    scheduler_config = config["scheduler"]
    scheduler_name = scheduler_config.get("name")

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **scheduler_config["params"])


def get_criterion(config: dict):
    loss_config = config["loss"]
    loss_name = loss_config["name"]
    loss_params = {} if loss_config.get("params") is None else loss_config.get(
        "params")

    if hasattr(nn, loss_name):
        criterion = nn.__getattribute__(loss_name)(**loss_params)
    else:
        criterion_cls = globals().get(loss_name)
        if criterion_cls is not None:
            criterion = criterion_cls(**loss_params)
        else:
            raise NotImplementedError

    return criterion


def get_split(config: dict):
    split_config = config["split"]
    name = split_config["name"]

    if hasattr(sms, name):
        return sms.__getattribute__(name)(**split_config["params"])
    else:
        return MultilabelStratifiedKFold(**split_config["params"])


def get_loader(df: pd.DataFrame,
               datadir: Path,
               config: dict,
               phase: str,
               event_level_labels=None,
               calltype_labels=None):
    dataset_config = config["dataset"]
    if dataset_config["name"] == "PANNsDataset":
        transforms = get_transforms(config, phase)
        loader_config = config["loader"][phase]

        dataset = datasets.PANNsDataset(
            df, datadir=datadir, transforms=transforms)
    elif dataset_config["name"] == "PANNsMultiLabelDataset":
        transforms = get_transforms(config, phase)
        loader_config = config["loader"][phase]
        period = dataset_config["params"][phase]["period"]
        dataset = datasets.PANNsMultiLabelDataset(
            df, datadir=datadir, transforms=transforms, period=period)
    elif dataset_config["name"] == "MultiChannelDataset":
        waveform_transforms = get_waveform_transforms(config, phase)
        spectrogram_transforms = get_spectrogram_transforms(config, phase)
        melspectrogram_parameters = dataset_config["params"]["melspectrogram_parameters"]
        pcen_parameters = dataset_config["params"]["pcen_parameters"]
        period = dataset_config["params"]["period"][phase]
        loader_config = config["loader"][phase]

        dataset = datasets.MultiChannelDataset(
            df,
            datadir=datadir,
            img_size=dataset_config["img_size"],
            waveform_transforms=waveform_transforms,
            spectrogram_transforms=spectrogram_transforms,
            melspectrogram_parameters=melspectrogram_parameters,
            pcen_parameters=pcen_parameters,
            period=period)
    elif dataset_config["name"] == "LabelCorrectionDataset":
        waveform_transforms = get_waveform_transforms(config, phase)
        spectrogram_transforms = get_spectrogram_transforms(config, phase)
        melspectrogram_parameters = dataset_config["params"]["melspectrogram_parameters"]
        pcen_parameters = dataset_config["params"]["pcen_parameters"]
        period = dataset_config["params"]["period"][phase]
        n_segments = dataset_config["params"]["n_segments"][phase]
        soft_label_dir = Path(dataset_config["params"]["soft_label_dir"])
        threshold = dataset_config["params"].get("threshold", 0.5)
        loader_config = config["loader"][phase]

        dataset = datasets.LabelCorrectionDataset(
            df,
            datadir=datadir,
            soft_label_dir=soft_label_dir,
            img_size=dataset_config["img_size"],
            waveform_transforms=waveform_transforms,
            spectrogram_transforms=spectrogram_transforms,
            melspectrogram_parameters=melspectrogram_parameters,
            pcen_parameters=pcen_parameters,
            period=period,
            n_segments=n_segments,
            threshold=threshold)
    else:
        raise NotImplementedError

    loader = data.DataLoader(dataset, **loader_config)
    return loader


def get_sed_inference_loader(df: pd.DataFrame, datadir: Path, config: dict):
    transforms = get_transforms(config, "train")
    if config["data"].get("denoised_audio_dir") is not None:
        denoised_audio_dir = Path(config["data"]["denoised_audio_dir"])
    else:
        denoised_audio_dir = None  # type: ignore
    if config.get("dataset") is None:
        dataset = datasets.PANNsSedDataset(
            df, datadir, transforms, denoised_audio_dir)
    elif config["dataset"]["name"] == "ChannelsSedDataset":
        melspectrogram_parameters = config["dataset"]["melspectrogram_parameters"]
        pcen_parameters = config["dataset"]["pcen_parameters"]
        period = config["dataset"]["period"]
        dataset = datasets.ChannelsSedDataset(
            df, datadir, transforms, denoised_audio_dir,
            melspectrogram_parameters,
            pcen_parameters, period)
    elif config["dataset"]["name"] == "NormalizedChannelsSedDataset":
        melspectrogram_parameters = config["dataset"]["melspectrogram_parameters"]
        pcen_parameters = config["dataset"]["pcen_parameters"]
        period = config["dataset"]["period"]
        dataset = datasets.NormalizedChannelsSedDataset(
            df, datadir, transforms, denoised_audio_dir,
            melspectrogram_parameters,
            pcen_parameters, period)
    loader = data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=8)
    return loader


def get_additional_metadata(config: dict):
    data_config = config["data"]
    train_e = pd.read_csv(data_config["train_extended_df_path"])
    if data_config.get("additional_labels_extended") is not None:
        with open(data_config["additional_labels_extended"]) as f:
            additional_labels = json.load(f)

        INV_NAME2CODE = {v: k for k, v in datasets.NAME2CODE.items()}
        for filename in additional_labels:
            labels = additional_labels[filename]
            labels = [INV_NAME2CODE[name] for name in labels]
            row_idx = train_e.query(f"resampled_filename == '{filename}'").index.values[0]
            train_e.loc[row_idx, "secondary_labels"] = str(labels)

    return train_e


def get_metadata(config: dict):
    data_config = config["data"]
    with open(data_config["train_skip"]) as f:
        skip_rows = f.readlines()

    train = pd.read_csv(data_config["train_df_path"])
    audio_path = Path(data_config["train_audio_path"])

    for row in skip_rows:
        row = row.replace("\n", "")
        ebird_code = row.split("/")[1]
        filename = row.split("/")[2]
        train = train[~((train["ebird_code"] == ebird_code) &
                        (train["filename"] == filename))]
        train = train.reset_index(drop=True)

    if data_config.get("additional_labels") is not None:
        with open(data_config["additional_labels"]) as f:
            additional_labels = json.load(f)

        INV_NAME2CODE = {v: k for k, v in datasets.NAME2CODE.items()}
        for filename in additional_labels:
            labels = additional_labels[filename]
            labels = [INV_NAME2CODE[name] for name in labels]
            row_idx = train.query(f"resampled_filename == '{filename}'").index.values[0]
            train.loc[row_idx, "secondary_labels"] = str(labels)

    if data_config.get("north_america", False):
        train = train[train.country.isin(["United States", "Canada", "Mexico"])].reset_index(drop=True)

    if data_config.get("use_extended", False) and data_config["train_extended_df_path"] is not None:
        train_e = get_additional_metadata(config)
        train_e.type = train_e.type.fillna("call")
        train_columns = set(train.columns)
        train_e_columns = set(train_e.columns)
        columns = list(train_columns.intersection(train_e_columns))
        train = train[columns]
        train_e = train_e[columns]
        train = pd.concat([train, train_e], axis=0, sort=False).reset_index(drop=True)

    return train, audio_path


def get_event_level_labels(config: dict):
    data_config = config["data"]

    event_level_labels = pd.read_csv(data_config["event_level_labels"])
    return event_level_labels
