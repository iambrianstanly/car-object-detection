import argparse
import yaml
from src.utils import set_seed
from src.data_loader import get_dataloaders
from src.model import get_model
from src.train import train_model


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()

    print(args.config)

    config_path = args.config


    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)


    set_seed(config["seed"])

    if config["mode"] == "train":

        train_ds, valid_ds = get_dataloaders(config["data"])

        model = get_model(config["model"])

        train_model(model, train_ds, valid_ds, config["train"])      



if __name__ == "__main__":
    main()


