import logging
from pathlib import Path


def setup_logger(log_dir: str | Path) -> logging.Logger:
    """
    Configure console and file logging.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("SegFormer")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(
        log_dir / "train.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

from datetime import datetime
from pathlib import Path
import shutil


def create_experiment_dir(config_path: str | Path) -> Path:
    """
    Create a unique directory for an experiment.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    experiment_dir = Path("outputs/experiments") / timestamp

    experiment_dir.mkdir(parents=True, exist_ok=False)

    # Save a copy of the configuration
    shutil.copy2(
        config_path,
        experiment_dir / "config.yaml",
    )

    (experiment_dir / "checkpoints").mkdir()

    return experiment_dir