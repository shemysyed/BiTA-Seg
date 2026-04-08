import torch
from src.models.vit_ccattn import BoundaryAwareViT
from src.evaluation.evaluator import dice
from src.data.dataset_loader import ImageMaskDataset
from torch.utils.data import DataLoader


def main(checkpoint):
    model = BoundaryAwareViT(img_size=256, patch_size=16, embed_dim=256, depth=6, num_heads=8)
    model.load_state_dict(torch.load(checkpoint))
    model.cuda()
    model.eval()

    ds = ImageMaskDataset("data/")
    dl = DataLoader(ds, batch_size=1)

    scores = []
    with torch.no_grad():
        for img, mask in dl:
            img, mask = img.cuda(), mask.cuda()
            pred = torch.sigmoid(model(img))
            scores.append(dice(pred, mask).item())

    print("Mean Dice:", sum(scores) / len(scores))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    args = p.parse_args()
    main(args.checkpoint)
