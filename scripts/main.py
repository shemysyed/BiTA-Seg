import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

test_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0).float() / 255.0  # ✅ convert to float 0-1

        return image, mask

class CrissCrossAttention(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x, return_attn=False):
        B, N, D = x.shape
        G = int(N ** 0.5)
        x_2d = x.view(B, G, G, D)

        # ---- Row Attention ----
        row_x = x_2d.permute(0, 2, 1, 3).reshape(B * G, G, D)
        row_out, row_attn = self.attn(row_x, row_x, row_x, need_weights=True)
        row_attn = row_attn.mean(1)

        # ---- Column Attention ----
        col_x = x_2d.reshape(B * G, G, D)
        col_out, col_attn = self.attn(col_x, col_x, col_x, need_weights=True)
        col_attn = col_attn.mean(1)

        out = (row_out.view(B, G, G, D).permute(0, 2, 1, 3) +
               col_out.view(B, G, G, D)).reshape(B, N, D)

        if return_attn:
            return out, row_attn, col_attn
        return out


class ViTBlockWithCCAttention(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim=2048):
        super(ViTBlockWithCCAttention, self).__init__()
        self.cc_attention = CrissCrossAttention(dim, num_heads)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.cc_attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=1, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, C * self.patch_size * self.patch_size)
        return self.proj(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    def forward(self, x):
        return x + self.pos_embed


class ViTCCSegmentation(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=1, embed_dim=256, num_heads=4, num_layers=6):
        super(ViTCCSegmentation, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, self.patch_embed.num_patches)
        self.blocks = nn.ModuleList([ViTBlockWithCCAttention(embed_dim, num_heads) for _ in range(num_layers)])
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1)
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.edge_proj = nn.Linear(1, embed_dim)

    def forward(self, x):
        B = x.shape[0]

        # ---- Patch Embedding ----
        patches = self.patch_embed(x)  # (B, N, D)
        x = self.pos_enc(patches)  # positional encoding

        # ---- Edge Tokens Storage ----
        edge_tokens = []

        # ---- Transformer Blocks ----
        for blk in self.blocks:
            # LayerNorm before attention (standard pre-norm)
            x_ln = blk.ln1(x)

            # Criss-Cross attention returns row/col attention
            attn_out, row_attn, col_attn = blk.cc_attention(x_ln, return_attn=True)

            # Standard ViT residual updates
            x = x + attn_out
            x = x + blk.ffn(blk.ln2(x))

            # ---- EDGE TOKEN GENERATION ----
            # row_attn & col_attn shapes: (B*G, G)
            div = torch.abs(row_attn - col_attn)  # (B*G, G)
            div = div.view(x.size(0), -1, 1)  # → (B, N, 1)

            # project to embedding dimension
            edge_tok = self.edge_proj(div)  # (B, N, D)

            edge_tokens.append(edge_tok)

        # ---- Combine Edge Tokens from All Layers ----
        edge_tokens = torch.stack(edge_tokens).mean(0)  # (B, N, D)

        # ---- Add Edge Embeddings to Patches ----
        x = x + edge_tokens

        # ---- Reshape to Feature Map ----
        grid_size = int(x.shape[1] ** 0.5)
        x = x.transpose(1, 2).view(B, self.embed_dim, grid_size, grid_size)

        # ---- Upsample to Image Size ----
        x = torch.nn.functional.interpolate(
            x, size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )

        # ---- Decoder ----
        x = self.decoder(x)

        return x



class DiceBCEBoundaryLoss(nn.Module):
    def __init__(self, smooth=1e-6, lambda_boundary=0.3):
        super(DiceBCEBoundaryLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_boundary = lambda_boundary

    def get_boundary(self, mask):

        laplacian_kernel = torch.tensor(
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]], dtype=torch.float32, device=mask.device
        ).unsqueeze(0).unsqueeze(0)
        boundary = torch.abs(F.conv2d(mask, laplacian_kernel, padding=1))
        boundary = (boundary > 0).float()
        return boundary

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        # Dice Loss
        intersection = (preds_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice

        # BCE Loss
        bce_loss = self.bce(preds, targets)

        # Boundary Loss
        pred_boundary = self.get_boundary(preds)
        target_boundary = self.get_boundary(targets)
        boundary_loss = F.l1_loss(pred_boundary, target_boundary)

        # Total
        total_loss = bce_loss + dice_loss + self.lambda_boundary * boundary_loss
        return total_loss


def dice_coeff(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    dice = (2. * intersection) / (preds.sum() + targets.sum() + 1e-6)
    return dice.item()

def segmentation_metrics(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float().cpu().numpy().flatten()
    targets = (targets > threshold).float().cpu().numpy().flatten()  # binarize masks

    # Avoid division or label mismatch issues
    if preds.max() == preds.min():  # all 0s or all 1s
        preds[0] = 1 - preds[0]  # make sure both labels exist
    if targets.max() == targets.min():
        targets[0] = 1 - targets[0]

    acc = accuracy_score(targets, preds)
    iou = jaccard_score(targets, preds, zero_division=1)
    prec = precision_score(targets, preds, zero_division=1)
    rec = recall_score(targets, preds, zero_division=1)
    return acc, iou, prec, rec


def visualize_predictions(model, dataloader, device, num_images=3):
    model.eval()
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            preds = torch.sigmoid(preds)
            preds = preds.cpu().numpy()
            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            for i in range(num_images):
                fig, ax = plt.subplots(1, 3, figsize=(10, 4))
                ax[0].imshow(images[i][0], cmap='gray'); ax[0].set_title("Input")
                ax[1].imshow(masks[i][0], cmap='gray'); ax[1].set_title("Ground Truth")
                ax[2].imshow(preds[i][0], cmap='gray'); ax[2].set_title("Prediction")
                for a in ax: a.axis('off')
                plt.show()
            break


train_dataset = SegmentationDataset(
    image_dir=r"C:\PROJECT FILES\Datasets\segmentation after split\train\images",
    mask_dir=r"C:\PROJECT FILES\Datasets\segmentation after split\train\masks",
    transform=train_transform
)

val_dataset = SegmentationDataset(
    image_dir=r"C:\PROJECT FILES\Datasets\segmentation after split\val\images",
    mask_dir=r"C:\PROJECT FILES\Datasets\segmentation after split\val\masks",
    transform=val_transform
)



train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

model = ViTCCSegmentation(img_size=256, patch_size=16, in_channels=1, embed_dim=256, num_heads=4, num_layers=6).to(device)
criterion = DiceBCEBoundaryLoss(lambda_boundary=0.3)

#optimizer = optim.AdamW(model.parameters(), lr=1e-4)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
num_epochs = 50
train_losses, val_losses, val_dice_scores = [], [], []
val_accs, val_ious, val_precs, val_recs = [], [], [], []
start = time.time()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    # Validation
    model.eval()
    val_loss, dice_score = 0, 0
    val_acc, val_iou, val_prec, val_rec = 0, 0, 0, 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)

            val_loss += criterion(preds, masks).item()
            dice_score += dice_coeff(preds, masks)

            acc, iou, prec, rec = segmentation_metrics(preds, masks)
            val_acc += acc
            val_iou += iou
            val_prec += prec
            val_rec += rec

    avg_val_loss = val_loss / len(val_loader)
    avg_dice = dice_score / len(val_loader)
    avg_acc = val_acc / len(val_loader)
    avg_iou = val_iou / len(val_loader)
    avg_prec = val_prec / len(val_loader)
    avg_rec = val_rec / len(val_loader)

    val_losses.append(avg_val_loss)
    val_dice_scores.append(avg_dice)
    val_accs.append(avg_acc)
    val_ious.append(avg_iou)
    val_precs.append(avg_prec)
    val_recs.append(avg_rec)

    print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f} | "
          f"Acc: {avg_acc:.4f} | Prec: {avg_prec:.4f} | Rec: {avg_rec:.4f}")
    # === Save final checkpoint after all epochs ===
    final_checkpoint_path = r"C:\PROJECT FILES\MODELS\Segmentation\model 50 rcvit-cnn segmentation\adtk_seg_new50.pth"

    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_dice_scores': val_dice_scores,
        'val_ious': val_ious,
        'val_accs': val_accs,
        'val_precs': val_precs,
        'val_recs': val_recs
    }, final_checkpoint_path)
end = time.time()

print(f"Total training time: {(end - start)/60:.2f} minutes")
print(f"✅ Final model and training state saved to: {final_checkpoint_path}")

print(f"✅ Final model and training state saved to: {final_checkpoint_path}")

plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.show()

plt.figure(figsize=(10,6))
plt.plot(val_dice_scores, label="Dice")
plt.plot(val_ious, label="IoU")
plt.plot(val_accs, label="Accuracy")
plt.plot(val_precs, label="Precision")
plt.plot(val_recs, label="Recall")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Segmentation Metrics Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

visualize_predictions(model, val_loader, device)

from tqdm import tqdm
test_dataset = SegmentationDataset(
    image_dir=r"C:\PROJECT FILES\Datasets\segmentation after split\test\images",
    mask_dir=r"C:\PROJECT FILES\Datasets\segmentation after split\test\masks",
    transform=test_transform
)

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


model.eval()
test_dice, test_iou, test_acc, test_prec, test_rec = 0, 0, 0, 0, 0
num_batches = len(test_loader)

with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Testing"):
        images, masks = images.to(device), masks.to(device)
        preds = model(images)

        # Metrics
        test_dice += dice_coeff(preds, masks)
        acc, iou, prec, rec = segmentation_metrics(preds, masks)
        test_acc += acc
        test_iou += iou
        test_prec += prec
        test_rec += rec

# Compute averages
test_dice /= num_batches
test_iou /= num_batches
test_acc /= num_batches
test_prec /= num_batches
test_rec /= num_batches


print(f"Dice: {test_dice:.4f}")
print(f"IoU: {test_iou:.4f}")
print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall: {test_rec:.4f}")

# Optional visualization
visualize_predictions(model, test_loader, device, num_images=3)

def extract_encoder_features(model, dataloader, device, num_images=4, patch_size=16):
    """
    Returns encoder patch embeddings, row/col attention maps, and corresponding patch labels (from masks).
    """
    model.eval()
    features, row_attn_maps, col_attn_maps, imgs, all_patch_labels = [], [], [], [], []

    with torch.no_grad():
        for images, masks in dataloader:  # ✅ get masks too
            images, masks = images.to(device), masks.to(device)
            imgs.append(images.cpu())

            # --- Encoder patch embeddings ---
            patches = model.patch_embed(images)
            x = model.pos_enc(patches)

            first_block = model.blocks[0].cc_attention
            B, N, D = x.shape
            grid_size = int(N ** 0.5)
            x_2d = x.view(B, grid_size, grid_size, D)

            # Row attention
            row_x = x_2d.permute(0, 2, 1, 3).reshape(B * grid_size, grid_size, D)
            _, row_attn = first_block.attn(row_x, row_x, row_x, need_weights=True)
            row_attn = row_attn.mean(1)
            row_attn_maps.append(row_attn.cpu())

            # Column attention
            col_x = x_2d.reshape(B * grid_size, grid_size, D)
            _, col_attn = first_block.attn(col_x, col_x, col_x, need_weights=True)
            col_attn = col_attn.mean(1)
            col_attn_maps.append(col_attn.cpu())

            # --- Patch-level labels (1 = lung, 0 = background) ---
            pooled = F.adaptive_avg_pool2d(masks, (grid_size, grid_size))
            patch_labels = (pooled > 0.5).float().view(B, -1)
            all_patch_labels.append(patch_labels.cpu())

            features.append(x.cpu().reshape(B * N, D))

            if len(features) >= num_images:
                break

    return torch.cat(features), row_attn_maps, col_attn_maps, torch.cat(imgs), torch.cat(all_patch_labels)


def visualize_tsne_with_labels(features, patch_labels):
    """
    t-SNE of encoder patch embeddings, colored by segmentation label.
    """
    print("Computing t-SNE (this may take a minute)...")
    X = features.numpy()
    y = patch_labels.numpy().flatten()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                c=y, cmap='coolwarm', s=6, alpha=0.7)
    plt.title("t-SNE of Encoder Patches\nRed = Lung, Blue = Background")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(label="Patch Type (0=Background, 1=Lung)")
    plt.grid(True)
    plt.show()
features, row_attn_maps, col_attn_maps, imgs, patch_labels = extract_encoder_features(model, val_loader, device)
visualize_tsne_with_labels(features, patch_labels)

# visualize row & column attentions side by side
def visualize_row_col_attention(row_attn, col_attn):
    attn_r = row_attn[0][0]
    attn_c = col_attn[0][0]

    len_r, len_c = attn_r.shape[0], attn_c.shape[0]
    g_r, g_c = int(len_r ** 0.5), int(len_c ** 0.5)
    attn_r = attn_r.reshape(g_r, g_r)
    attn_c = attn_c.reshape(g_c, g_c)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(attn_r, cmap='hot')
    axs[0].set_title(f'Row Attention ({g_r}×{g_r})')

    axs[1].imshow(attn_c, cmap='hot')
    axs[1].set_title(f'Column Attention ({g_c}×{g_c})')

    plt.show()
def visualize_combined_attention(row_attn, col_attn):
    attn_r = row_attn[0][0]
    attn_c = col_attn[0][0]
    len_r, len_c = attn_r.shape[0], attn_c.shape[0]
    g_r, g_c = int(len_r ** 0.5), int(len_c ** 0.5)

    attn_r = attn_r.reshape(g_r, g_r)
    attn_c = attn_c.reshape(g_c, g_c)

    combined = (attn_r + attn_c) / 2  # average row and column attention

    plt.figure(figsize=(6,6))
    plt.imshow(combined, cmap='hot', interpolation='nearest')
    plt.title(f"Combined Criss-Cross Attention ({g_r}×{g_r})")
    plt.colorbar()
    plt.show()

visualize_row_col_attention(row_attn_maps, col_attn_maps)
visualize_combined_attention(row_attn_maps, col_attn_maps)


