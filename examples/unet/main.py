""" Example Code of U-Net
"""
import shutil
from logging import getLogger
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
# import openpack_toolkit as optk
import openpack_torch as optorch
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from openpack_toolkit import \
    OPENPACK_WORKPROCESS_CLASSES as OPENPACK_OPERATION_CLASSES
from openpack_toolkit.codalab.workprocess_segmentation import (
    construct_submission_dict, eval_workprocess_segmentation_wrapper,
    make_submission_zipfile)

logger = getLogger(__name__)

# ----------------------------------------------------------------------


def cleanup_logdir(logdir: Path) -> None:
    """Remove files and directories in logdir.
    Keep files generated by hydra.

    Args:
        logdir (Path): _description_
    Todo:
        Register this function to optk.
    """
    logger.debug(f"clean up {logdir}")
    for path in logdir.iterdir():
        if "hydra" in path.name:
            continue
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=False)
    return


def save_training_results(log: Dict, logdir: Path) -> None:
    # -- Save Model Outputs --
    df = pd.concat(
        [
            pd.DataFrame(log["train"]),
            pd.DataFrame(log["val"]),
        ],
        axis=1,
    )
    df.index.name = "epoch"

    path = Path(logdir, "training_log.csv")
    df.to_csv(path, index=True)
    logger.debug(f"Save training logs to {path}")
    print(df)


# # # ----------------------------------------------------------------------
class OpenPackImuDataModule(optorch.data.OpenPackBaseDataModule):
    dataset_class = optorch.data.datasets.OpenPackImu

    def get_kwargs_for_datasets(self) -> Dict:
        imu_cfg = self.cfg.dataset.modality.imu
        kwargs = {
            "imu_nodes": imu_cfg.nodes,
            "use_acc": imu_cfg.use_acc,
            "use_gyro": imu_cfg.use_gyro,
            "use_quat": imu_cfg.use_quat,
            "window": self.cfg.train.window,
            "debug": self.cfg.debug,
        }
        return kwargs


class UNetLM(optorch.lightning.BaseLightningModule):

    def init_model(self, cfg: DictConfig) -> torch.nn.Module:
        imu_cfg = self.cfg.dataset.modality.imu
        num_nodes = len(imu_cfg.nodes)

        in_ch = 0
        if imu_cfg.use_acc:
            in_ch += num_nodes * 3
        if imu_cfg.use_gyro:
            in_ch += num_nodes * 3
        if imu_cfg.use_quat:
            in_ch += num_nodes * 4

        model = optorch.models.imu.UNet(
            in_ch,
            cfg.dataset.num_classes,
            depth=cfg.model.depth,
        )
        return model

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        x = batch["x"].to(device=self.device, dtype=torch.float)
        t = batch["t"].to(device=self.device, dtype=torch.long)
        y_hat = self(x).squeeze(3)

        loss = self.criterion(y_hat, t)
        acc = self.calc_accuracy(y_hat, t)
        return {"loss": loss, "acc": acc}

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        x = batch["x"].to(device=self.device, dtype=torch.float)
        t = batch["t"].to(device=self.device, dtype=torch.long)
        ts_unix = batch["ts"]

        y_hat = self(x).squeeze(3)

        outputs = dict(t=t, y=y_hat, unixtime=ts_unix)
        return outputs


# ----------------------------------------------------------------------


def train(cfg: DictConfig):
    device = torch.device("cuda")
    logdir = Path.cwd()
    logger.debug(f"logdir = {logdir}")
    cleanup_logdir(logdir)

    datamodule = OpenPackImuDataModule(cfg)
    plmodel = UNetLM(cfg)
    plmodel.to(dtype=torch.float, device=device)
    logger.info(plmodel)

    num_epoch = cfg.train.debug.epochs if cfg.debug else cfg.train.epochs

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=0,
        save_last=True,
        monitor=None,
    )

    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=num_epoch,
        logger=False,  # disable logging module
        default_root_dir=logdir,
        enable_progress_bar=False,  # disable progress bar
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )
    logger.debug(f"logdir = {logdir}")

    logger.info(f"Start training for {num_epoch} epochs.")
    trainer.fit(plmodel, datamodule)
    logger.info("Finish training!")

    logger.debug(f"logdir = {logdir}")
    save_training_results(plmodel.log, logdir)
    logger.debug(f"logdir = {logdir}")


def test(cfg: DictConfig, mode: str = "test"):
    assert mode in ("test", "submission")
    logger.debug(f"test() function is called with mode={mode}.")

    device = torch.device("cuda")
    logdir = Path(cfg.volume.logdir.rootdir)

    datamodule = OpenPackImuDataModule(cfg)
    datamodule.setup(mode)

    ckpt_path = Path(logdir, "checkpoints", "last.ckpt")
    logger.info(f"load checkpoint from {ckpt_path}")
    plmodel = UNetLM.load_from_checkpoint(ckpt_path, cfg=cfg)
    plmodel.to(dtype=torch.float, device=device)

    # num_epoch = cfg.train.debug.epochs if cfg.debug else cfg.train.epochs
    trainer = pl.Trainer(
        gpus=[0],
        logger=False,  # disable logging module
        default_root_dir=None,
        enable_progress_bar=False,  # disable progress bar
        enable_checkpointing=False,  # does not save model check points
    )

    if mode == "test":
        dataloaders = datamodule.test_dataloader()
        split = cfg.dataset.split.test
    elif mode == "submission":
        dataloaders = datamodule.submission_dataloader()
        split = cfg.dataset.split.submission
    outputs = dict()
    for i, dataloader in enumerate(dataloaders):
        user, session = split[i]
        logger.info(f"test on U{user:0=4}-S{session:0=4}")

        trainer.test(plmodel, dataloader)

        # save model outputs
        pred_dir = Path(
            cfg.volume.logdir.predict.format(user=user, session=session)
        )
        pred_dir.mkdir(parents=True, exist_ok=True)

        for key, arr in plmodel.test_results.items():
            path = Path(pred_dir, f"{key}.npy")
            np.save(path, arr)
            logger.info(f"save {key}[shape={arr.shape}] to {path}")

        key = f"U{user:0=4}-S{session:0=4}"
        outputs[key] = {
            "y": plmodel.test_results.get("y"),
            "unixtime": plmodel.test_results.get("unixtime"),
        }
        if mode == "test":
            outputs[key].update({
                "t_idx": plmodel.test_results.get("t"),
            })

    if mode == "test":
        # save performance summary
        df_summary = eval_workprocess_segmentation_wrapper(
            outputs, OPENPACK_OPERATION_CLASSES,
        )
        path = Path(cfg.volume.logdir.summary)
        df_summary.to_csv(path, index=False)
        logger.info(f"df_summary:\n{df_summary}")
    elif mode == "submission":
        # make submission file
        submission_dict = construct_submission_dict(
            outputs, OPENPACK_OPERATION_CLASSES)
        make_submission_zipfile(submission_dict, logdir)


@ hydra.main(version_base=None, config_path="../../configs",
             config_name="operation-segmentation-unet.yaml")
def main(cfg: DictConfig):
    print("===== Params =====")
    print(OmegaConf.to_yaml(cfg))
    print("==================")
    pl.seed_everything(42, workers=True)
    logger.debug(f"check = {Path.cwd()}")
    print(f'Orig working directory : {hydra.utils.get_original_cwd()}')
    print(cfg.volume.logdir.rootdir)

    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode in ("test", "submission"):
        test(cfg, mode=cfg.mode)
    else:
        raise ValueError(f"unknown mode [cfg.mode={cfg.mode}]")


if __name__ == "__main__":
    main()
