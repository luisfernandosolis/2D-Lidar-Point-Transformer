import argparse
import json
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from model.LidarPointTransformer import LidarPointTransformer

class LidarRegionPointDataset(Dataset):
    def __init__(self, json_path, num_classes=None):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.index = []
        self.num_classes = num_classes
        # flatten per scene: for each scene create per-region mapping and label points as class id
        for scene in self.data:
            lidar_path = scene.get("lidar_file_path", "")
            try:
                ld = pd.read_csv(lidar_path)
            except Exception as e:
                continue
            for region in scene.get("regions", []):
                ids = region.get("lidar_cluster_indices", [])
                if len(ids) == 0:
                    continue
                pts = ld[ld["id"].isin(ids)]
                distances = np.abs(pts["distance"].values)
                angles = pts["angle"].values
                if np.nanmax(np.abs(angles)) > 2*np.pi + 1e-3:
                    angles = np.deg2rad(angles)
                x = distances * np.cos(angles)
                y = distances * np.sin(angles)
                xy = np.stack([x, y], axis=-1)
                label = region.get("semantic_label_clean", region.get("semantic_label", None))
                if label is None:
                    try:
                        label_int = int(''.join(filter(str.isdigit, region["region_id"])))
                    except:
                        label_int = 0
                else:
                    label_int = 0
                sample = {"points": xy.astype(np.float32), "label": int(label_int)}
                self.index.append(sample)
        if self.num_classes is None:
            maxl = max([s["label"] for s in self.index]) if self.index else 0
            self.num_classes = maxl + 1

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s = self.index[idx]
        return s["points"], s["label"]

def collate_fn(batch):
    max_pts = max([b[0].shape[0] for b in batch])
    B = len(batch)
    x = np.zeros((B, max_pts, 2), dtype=np.float32)
    mask = np.zeros((B, max_pts), dtype=np.bool_)
    y = np.zeros((B,), dtype=np.int64)
    for i, (pts, label) in enumerate(batch):
        n = pts.shape[0]
        x[i, :n] = pts
        mask[i, :n] = True
        y[i] = label
    return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(y)

def train_loop(args):
    dataset = LidarRegionPointDataset(args.dataset_json)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model = LidarPointTransformer(in_dim=2, embed_dim=128, n_heads=4, n_layers=2, num_classes=max(2, dataset.num_classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        losses = []
        total = 0
        correct = 0
        for x, mask, y in dl:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            logits = model(x, mask)
            logits_mean = logits.mean(dim=1)  # B x C
            loss = criterion(logits_mean, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            preds = logits_mean.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
        avg_loss = sum(losses)/len(losses) if losses else 0
        acc = correct/total if total>0 else 0
        print(f"Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f} acc={acc:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "lidar_point_transformer.pth"))
    print("Saved checkpoint to", os.path.join(args.out_dir, "lidar_point_transformer.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train the LidarPointTransformer model")
    parser.add_argument("--dataset-json", type=str, default="final_processed_dataset.json")
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()
    train_loop(args)
