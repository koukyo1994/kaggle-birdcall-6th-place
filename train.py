import warnings

import src.callbacks as clb
import src.configuration as C
import src.models as models
import src.utils as utils

from pathlib import Path

from catalyst.dl import SupervisedRunner


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

    if config["data"].get("event_level_labels") is not None:
        event_level_labels = C.get_event_level_labels(config)
    else:
        event_level_labels = None

    for i, (trn_idx, val_idx) in enumerate(
            splitter.split(df, y=df["ebird_code"])):
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
        callbacks = clb.get_callbacks(config)

        runner = SupervisedRunner(
            device=device,
            input_key=global_params["input_key"],
            input_target_key=global_params["input_target_key"])
        runner.train(
            model=model,
            criterion=criterion,
            loaders=loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=global_params["num_epochs"],
            verbose=True,
            logdir=output_dir / f"fold{i}",
            callbacks=callbacks,
            main_metric=global_params["main_metric"],
            minimize_metric=global_params["minimize_metric"])
