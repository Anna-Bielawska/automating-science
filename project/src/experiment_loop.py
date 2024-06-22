import json
import numpy as np
import torch
from torch import nn
from datetime import datetime

from config.main_config import MainConfig
from src.loops.base_loop import BaseLoop
from src.loops.mutate_loop import MutateLoop
from src.loops.loop_registry import create_loop
import src.models.model_registry
from src.utils.training import (
    train_epoch,
    validate_epoch,
    MAELoss,
    get_data_loaders,
    EarlyStopper,
    set_seed,
)
import logging
from pathlib import Path
from src.utils.dataset import SmallZINC
from src.utils.molecules import LeadCompound
import src.models
import copy
from config.loops import MutateLoopParams

logger = logging.getLogger(__name__)


def run(loop: BaseLoop, budget: int = 1000, steps: int = 10):
    """Run the loop for a given number of steps and with a given budget.

    Args:
        loop (BaseLoop): Loop object to test.
        budget (int, optional): Budget of the molecules for the loop. Defaults to 1000.
        steps (int, optional): Number of steps to run the loop. Defaults to 10.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    Path(loop.base_dir).mkdir(parents=True, exist_ok=True)

    metrics = []
    all_result: list[LeadCompound] = []
    budget_per_step = budget // steps
    assert budget % steps == 0  # for simplicity

    for step in range(steps):
        logger.info(f"Step {step}")

        candidates = loop.propose_candidates(budget_per_step)
        loop.test_in_lab_and_save(candidates)
        result: list[LeadCompound] = loop.load(iteration_id=step)
        all_result += result
        all_result_sorted = sorted(all_result, key=lambda x: x.activity, reverse=True)

        metrics.append(
            {
                "step": step,
                "top10": np.mean([x.activity for x in all_result_sorted[:10]]),
                "top10_synth": np.mean([x.synth_score for x in all_result_sorted[:10]]),
            }
        )

    return metrics


def pretrain_loop(
    cfg: MainConfig,
    hydra_output_dir: Path,
    model: nn.Module,
    dataset_space: SmallZINC,
) -> None:
    device = torch.device(cfg._device)
    pretrain_cfg = cfg.pretraining
    model.to(device)

    compounds_sample = [
        LeadCompound(smiles=smile, synth_score=None, activity=None)
        for smile in dataset_space.try_sample(pretrain_cfg.compounds_budget)
    ]

    mutate_loop = MutateLoop(
        loop_params=MutateLoopParams(),
        base_dir=hydra_output_dir / "pretraining/molecules",
        initial_dataset=compounds_sample,
    )

    # gather high activity compounds for pretraining
    loop_run_metrics = run(
        mutate_loop,
        budget=pretrain_cfg.compounds_budget,
        steps=pretrain_cfg.loop_steps,
    )
    compounds = mutate_loop.load()

    # # save metrics as json file
    with open(hydra_output_dir / "pretraining/mutate_loop_metrics.json", "w") as f:
        json.dump(loop_run_metrics, f)

    optimizer = torch.optim.Adam(model.parameters(), 5e-4)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopper(**pretrain_cfg.early_stopping)

    train_dl, valid_dl = get_data_loaders(
        compounds,
        pretrain_cfg.split_ratio,
        pretrain_cfg.batch_size,
        pretrain_cfg.dataset_path,
    )

    best_model = model.state_dict()
    best_loss = float("inf")

    # pretraining loop
    for epoch in range(pretrain_cfg.num_epochs):
        logger.info(f"Epoch {epoch + 1}")
        train_loss = train_epoch(model, train_dl, optimizer, criterion, device)
        logger.info(f"Train loss: {train_loss}")

        valid_loss = validate_epoch(model, valid_dl, criterion, device)

        logger.info(f"Validation loss: {valid_loss}")

        if early_stopping.enabled and early_stopping.check_stop(valid_loss):
            logger.info("Early stopping triggered.")
            break

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = model.state_dict()

    model.load_state_dict(best_model)

    if pretrain_cfg.save_model:
        torch.save(model.state_dict(), hydra_output_dir / "pretrained_model.pth")


def train_loop(
    cfg: MainConfig,
    hydra_output_dir: Path,
    repeat: int,
    model: nn.Module,
    dataset_space: SmallZINC,
) -> None:
    model = copy.deepcopy(model)

    device = torch.device(cfg._device)
    train_cfg = cfg.training
    model = model.to(device)

    compounds_sample = [
        LeadCompound(smiles=smile, synth_score=None, activity=None)
        for smile in dataset_space.try_sample(train_cfg.compounds_budget)
    ]

    loop_name = f"{train_cfg.loop.name}"
    mutate_loop = MutateLoop(
        loop_params=MutateLoopParams(),
        base_dir=hydra_output_dir / f"repeat_{repeat}/molecules",
        initial_dataset=compounds_sample,
    )

    loop = create_loop(
        loop_name,
        loop_params=train_cfg.loop.params,
        train_cfg=train_cfg,
        base_dir=hydra_output_dir / f"repeat_{repeat}/molecules",
        base_loop=mutate_loop,
        model=model,
        device=device,
    )

    loop_run_metrics = run(
        loop,
        budget=cfg.training.compounds_budget,
        steps=train_cfg.loop_steps,
    )

    # save metrics as json file
    with open(hydra_output_dir / f"repeat_{repeat}/{loop_name}_metrics.json", "w") as f:
        json.dump(loop_run_metrics, f)


def start_experiment_loop(cfg: MainConfig, hydra_output_dir: Path) -> None:
    """Starts the main experiment loop.

    Args:
        cfg (MainConfig): Hydra config object with all the settings.
        hydra_output_dir (Path): Path to the output directory generated by Hydra.
    """
    set_seed(cfg.seed)
    experiment_start_time = datetime.now().strftime("%b%d_%H-%M-%S")

    dataset_space = SmallZINC(seed=cfg.seed)

    model = src.models.model_registry.create_model(cfg.model.name, **cfg.model.params)

    if cfg.pretraining.enabled:
        pretrain_loop(cfg, hydra_output_dir, model, dataset_space)

    for repeat in range(cfg._repeat):
        logger.info(f"Repeat {repeat + 1}")
        set_seed(cfg.seed + repeat)
        train_loop(cfg, hydra_output_dir, repeat, model, dataset_space)
