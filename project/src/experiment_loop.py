import copy
import json
import logging
from pathlib import Path

import numpy as np
import src.models
import src.models.model_registry
import src.utils
import src.utils.training
import torch
from config.loops import MutateLoopParams
from config.main_config import MainConfig
from src.embeddings.embedding_registry import create_embedding
from src.loops.base_loop import BaseLoop
from src.loops.loop_registry import create_loop
from src.loops.mutate_loop import MutateLoop
from src.utils.dataset import SmallZINC
from src.utils.molecules import LeadCompound
from torch import nn

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
        all_result_sorted = sorted(
            all_result, key=lambda x: (x.activity, x.smiles), reverse=True
        )

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
    """
    Pretraining loop.
    
    Args:
        cfg (MainConfig): Hydra config object with all the settings.
        hydra_output_dir (Path): Path to the output directory generated by Hydra.
        model (nn.Module): Model to pretrain.
        dataset_space (SmallZINC): Dataset space to sample molecules from.
    """
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
    early_stopping = src.utils.training.EarlyStopper(**pretrain_cfg.early_stopping)
    embedding = create_embedding(pretrain_cfg.embedding)

    train_dl, valid_dl = src.utils.training.get_data_loaders(
        compounds,
        embedding,
        pretrain_cfg.split_ratio,
        pretrain_cfg.batch_size,
    )

    best_model = model.state_dict()
    best_loss = float("inf")

    # pretraining loop
    for epoch in range(pretrain_cfg.num_epochs):
        train_loss = src.utils.training.train_epoch(
            model, train_dl, optimizer, criterion, device
        )

        valid_loss = src.utils.training.validate_epoch(
            model, valid_dl, criterion, device
        )

        logger.info(
            f"Epoch {epoch + 1}: Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}"
        )

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
    """
    Training loop.

    Args:
        cfg (MainConfig): Hydra config object with all the settings.
        hydra_output_dir (Path): Path to the output directory generated by Hydra.
        repeat (int): Repeat number.
        model (nn.Module): Model to train.
        dataset_space (SmallZINC): Dataset space to sample molecules from.
    """
    model = copy.deepcopy(model)

    device = torch.device(cfg._device)
    train_cfg = cfg.training
    model = model.to(device)

    compounds_sample = [
        LeadCompound(smiles=smile, synth_score=None, activity=None)
        for smile in dataset_space.try_sample(train_cfg.compounds_budget)
    ]

    mutate_loop = MutateLoop(
        loop_params=MutateLoopParams(),
        base_dir=hydra_output_dir / f"repeat_{repeat}/molecules",
        initial_dataset=compounds_sample,
    )

    loop = create_loop(
        loop_cfg=train_cfg.loop,
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

    if train_cfg.save_model:
        torch.save(model.state_dict(), hydra_output_dir / f"repeat_{repeat}/model.pth")

    return loop_run_metrics


def start_experiment_loop(cfg: MainConfig, hydra_output_dir: Path) -> None:
    """Starts the main experiment loop.

    Args:
        cfg (MainConfig): Hydra config object with all the settings.
        hydra_output_dir (Path): Path to the output directory generated by Hydra.
    """
    src.utils.training.set_seed(cfg.seed)

    dataset_space = SmallZINC(seed=cfg.seed)

    model = src.models.model_registry.create_model(cfg.model)

    if cfg.pretraining.enabled:
        pretrain_loop(cfg, hydra_output_dir, model, dataset_space)

    train_loop_metrics = []
    for repeat in range(cfg._repeat):
        logger.info(f"Repeat {repeat + 1}")
        src.utils.training.set_seed(cfg.seed + repeat)
        metrics = train_loop(cfg, hydra_output_dir, repeat, model, dataset_space)
        train_loop_metrics.append(metrics)


    with open(hydra_output_dir / f"{cfg.training.loop.name}_metrics.json", "w") as f:
        json.dump(train_loop_metrics, f)


