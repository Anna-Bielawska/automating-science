import hydra
import logging
from pathlib import Path

from omegaconf import OmegaConf

from config.main_config import MainConfig
from src.experiment_loop import start_experiment_loop

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main_config", version_base="1.3")
def main(cfg: MainConfig) -> None:
    """Main function for the pruning entry point

    Args:
        cfg (MainConfig): Hydra config object with all the settings. (Located in config/main_config.py)
    """
    logger.setLevel(cfg._logging_level)

    logger.info(OmegaConf.to_yaml(cfg))
    hydra_output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    logger.info(f"Hydra output directory: {hydra_output_dir}")

    start_experiment_loop(cfg, hydra_output_dir)


if __name__ == "__main__":
    main()
