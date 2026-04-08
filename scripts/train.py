import yaml
import torch
from torch.utils.data import DataLoader

from src.models.vit_ccattn import BoundaryAwareViT
from src.data.dataset_loader import ImageMaskDataset
from src.training.trainer import Trainer
from src.training.losses import DiceBCELoss


def main(config_path):
    cfg = yaml.safe_load(open(config_path))

    ds = ImageMaskDataset(cfg["data"]["root"])
    dl = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True)

    model = BoundaryAwareViT(**cfg["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    criterion = DiceBCELoss()
    trainer = Trainer(model, optimizer, criterion, device="cuda")

    for epoch in range(cfg["training"]["epochs"]):
        loss = trainer.train_epoch(dl)
        print(f"Epoch {epoch}: {loss:.4f}")

        torch.save(model.state_dict(), f"{cfg['training']['save_dir']}/epoch{epoch}.pth")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.config)
