import argparse
import logging
import os
import sys

from reasoning_with_art.config import load_config

def seed_everything(seed=42):
    # 1. Base Python built-in random module
    import os
    import random
    import numpy as np
    import torch
    random.seed(seed)
    
    # 2. Python hashing environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. NumPy library
    np.random.seed(seed)
    
    # 4. PyTorch CPU and CUDA (all GPUs)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 5. PyTorch deterministic operations (can impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Run reasoning-with-art experiments")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../../configs/default.yaml"),
        help="Path to experiment config YAML",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    config = load_config(args.config)

    if config.gpu_ids is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in config.gpu_ids)
        logging.getLogger(__name__).info(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    seed_everything()

    # Must happen AFTER CUDA_VISIBLE_DEVICES is set, otherwise contexts leak onto every GPU.
    from reasoning_with_art.pipeline import print_summary, run

    results = run(config)
    print_summary(results)
