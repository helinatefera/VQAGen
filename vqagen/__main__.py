from omegaconf import OmegaConf
from .train import Trainer
def main():
    cfg = OmegaConf.load("configs/defualt.yml")
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
