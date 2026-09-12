"""Entry point: python -m src.train --config experiments/configs/default.yaml"""
import argparse

import yaml


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    raise NotImplementedError(f"Training not implemented yet. Loaded config: {config['model']['name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/default.yaml")
    main(parser.parse_args().config)
