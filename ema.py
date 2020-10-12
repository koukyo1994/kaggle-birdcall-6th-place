import time
import warnings

import numpy as np
import torch
import torch.nn as nn

import src.configuration as C
import src.models as models
import src.utils as utils

from copy import deepcopy
from pathlib import Path

from fastprogress import progress_bar
from sklearn.metrics import average_precision_score, f1_score


class AveragedModel(nn.Module):
    def __init__(self, model, device=None, avg_fn=None):
        super().__init__()
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer("n_averaged",
                             torch.tensor(0, dtype=torch.long, device=device))
        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + \
                    (model_parameter - averaged_model_parameter) / (num_averaged + 1)
        self.avg_fn = avg_fn

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model):
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_, self.n_averaged.to(device)))
            self.n_averaged += 1


def update_bn(loader, model, device=None, input_key=""):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Arguments:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.
    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)
    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if isinstance(input, dict):
            input = input[input_key]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


def train_one_epoch(model,
                    ema_model,
                    dataloader,
                    optimizer,
                    scheduler,
                    criterion,
                    device,
                    n=10,
                    input_key="image",
                    input_target_key="targets"):
    avg_loss = 0.0
    model.train()
    preds = []
    targs = []
    cnt = n
    for step, batch in enumerate(progress_bar(dataloader)):
        cnt -= 1
        x = batch[input_key].to(device)
        y = batch[input_target_key].to(device).float()

        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item() / len(dataloader)
        if cnt == 0:
            ema_model.update_parameters(model)
            cnt = n

        clipwise_output = outputs["clipwise_output"].detach().cpu().numpy()
        target = y.detach().cpu().numpy()

        preds.append(clipwise_output)
        targs.append(target)

    update_bn(dataloader, ema_model, device=device, input_key=input_key)

    scheduler.step()

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targs, axis=0)
    return avg_loss, y_pred, y_true


def eval_one_epoch(model,
                   dataloader,
                   criterion,
                   device,
                   input_key="image",
                   input_target_key="targets"):
    avg_loss = 0.0
    model.eval()
    preds = []
    targs = []
    for step, batch in enumerate(progress_bar(dataloader)):
        with torch.no_grad():
            x = batch[input_key].to(device)
            y = batch[input_target_key].to(device).float()

            outputs = model(x)
            loss = criterion(outputs, y).detach()

            avg_loss += loss.item() / len(dataloader)

            clipwise_output = outputs["clipwise_output"].detach().cpu().numpy()
            target = y.detach().cpu().numpy()

            preds.append(clipwise_output)
            targs.append(target)

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targs, axis=0)
    return avg_loss, y_pred, y_true


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold=0.5):
    mAP = average_precision_score(y_true, y_pred, average=None)
    mAP = np.nan_to_num(mAP).mean()

    classwise_f1s = []
    for i in range(len(y_true[0])):
        class_i_pred = y_pred[:, i] > threshold
        class_i_targ = y_true[:, i]
        if class_i_targ.sum() == 0 and class_i_pred.sum() == 0:
            classwise_f1s.append(1.0)
        else:
            classwise_f1s.append(f1_score(y_true=class_i_targ, y_pred=class_i_pred))

    classwise_f1 = np.mean(classwise_f1s)

    y_pred_thresholded = (y_pred > threshold).astype(int)
    sample_f1 = f1_score(y_true=y_true, y_pred=y_pred_thresholded, average="samples")
    return mAP, classwise_f1, sample_f1


def save_model(model, logdir: Path, filename: str):
    state_dict = {}
    state_dict["model_state_dict"] = model.state_dict()

    weights_path = logdir / filename
    with open(weights_path, "wb") as f:
        torch.save(state_dict, f)


def save_best_model(model, logdir, filename, metric: float, prev_metric: float):
    if metric > prev_metric:
        save_model(model, logdir, filename)
        return metric
    else:
        return prev_metric


def train(model,
          ema_model,
          dataloaders,
          optimizer,
          scheduler,
          criterion,
          device,
          logdir: Path,
          logger,
          n=10,
          main_metric="sample_f1",
          epochs=75,
          input_key="image",
          input_target_key="targets"):
    train_metrics = {}
    eval_metrics = {}
    best_metric = -np.inf
    for epoch in range(epochs):
        t0 = time.time()
        epoch += 1
        logger.info("=" * 20)
        logger.info(f"Epoch [{epoch}/{epochs}]:")
        logger.info("=" * 20)

        logger.info("Train")

        avg_loss, y_pred, y_true = train_one_epoch(
            model=model,
            ema_model=ema_model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            n=n,
            input_key=input_key,
            input_target_key=input_target_key)
        mAP, classwise_f1, sample_f1 = calc_metrics(y_true, y_pred)
        train_metrics["loss"] = avg_loss
        train_metrics["mAP"] = mAP
        train_metrics["classwise_f1"] = classwise_f1
        train_metrics["sample_f1"] = sample_f1

        if len(dataloaders) == 1:
            val_dataloader = dataloaders["train"]
        else:
            val_dataloader = dataloaders["valid"]

        logger.info("Valid")

        avg_loss, y_pred, y_true = eval_one_epoch(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
            input_key=input_key,
            input_target_key=input_target_key)
        mAP, classwise_f1, sample_f1 = calc_metrics(y_true, y_pred)
        eval_metrics["loss"] = avg_loss
        eval_metrics["mAP"] = mAP
        eval_metrics["classwise_f1"] = classwise_f1
        eval_metrics["sample_f1"] = sample_f1

        logger.info("EMA")

        avg_loss, y_pred, y_true = eval_one_epoch(
            model=ema_model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
            input_key=input_key,
            input_target_key=input_target_key)
        mAP, classwise_f1, sample_f1 = calc_metrics(y_true, y_pred)
        eval_metrics["EMA_loss"] = avg_loss
        eval_metrics["EMA_mAP"] = mAP
        eval_metrics["EMA_classwise_f1"] = classwise_f1
        eval_metrics["EMA_sample_f1"] = sample_f1

        logger.info("#" * 20)
        logger.info("Train metrics")
        for key, value in train_metrics.items():
            logger.info(f"{key}: {value:.5f}")

        logger.info("Valid metrics")
        for key, value in eval_metrics.items():
            logger.info(f"{key}: {value:.5f}")

        logger.info("#" * 20)
        best_metric = save_best_model(
            model, logdir, "best.pth",
            metric=eval_metrics[main_metric], prev_metric=best_metric)

        save_model(ema_model, logdir, "ema.pth")
        elapsed_sec = time.time() - t0
        elapsed_min = int(elapsed_sec // 60)
        elapsed_sec = elapsed_sec % 60
        logger.info(f"Elapsed time: {elapsed_min}min {elapsed_sec:.4f}seconds.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    args = utils.get_parser().parse_args()
    config = utils.load_config(args.config)

    global_params = config["globals"]

    output_dir = Path(global_params["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = utils.get_logger(output_dir / "output.log")

    utils.set_seed(global_params["seed"])
    device = C.get_device(global_params["device"])

    df, datadir = C.get_metadata(config)
    splitter = C.get_split(config)

    calltype_labels = C.get_calltype_labels(df)

    if config["data"].get("event_level_labels") is not None:
        event_level_labels = C.get_event_level_labels(config)
    else:
        event_level_labels = None

    if "Multilabel" in config["split"]["name"]:
        y = calltype_labels
    else:
        y = df["ebird_code"]
    for i, (trn_idx, val_idx) in enumerate(
            splitter.split(df, y=y)):
        if i not in global_params["folds"]:
            continue
        logger.info("=" * 20)
        logger.info(f"Fold {i}")
        logger.info("=" * 20)

        trn_df = df.loc[trn_idx, :].reset_index(drop=True)
        val_df = df.loc[val_idx, :].reset_index(drop=True)

        loaders = {
            phase: C.get_loader(df_, datadir, config, phase, event_level_labels)
            for df_, phase in zip([trn_df, val_df], ["train", "valid"])
        }
        model = models.get_model(config).to(device)
        criterion = C.get_criterion(config).to(device)
        optimizer = C.get_optimizer(model, config)
        scheduler = C.get_scheduler(optimizer, config)

        ema_model = AveragedModel(
            model,
            avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged:
                0.1 * averaged_model_parameter + 0.9 * model_parameter)

        (output_dir / f"fold{i}").mkdir(exist_ok=True, parents=True)

        train(model=model,
              ema_model=ema_model,
              dataloaders=loaders,
              optimizer=optimizer,
              scheduler=scheduler,
              criterion=criterion,
              device=device,
              logdir=output_dir / f"fold{i}",
              logger=logger,
              n=10,
              main_metric=global_params["main_metric"],
              epochs=global_params["num_epochs"],
              input_key=global_params["input_key"],
              input_target_key=global_params["input_target_key"])
