import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class DiskCropDataset(Dataset):
    def __init__(self, image_dir, coord_csv_path, transform=None, crop_size=64):
        self.image_dir = image_dir
        self.data = pd.read_csv(coord_csv_path)
        self.transform = transform
        self.crop_size = crop_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("L")
        W, H = image.size

        crops = []
        for i in range(1, 6):
            # 1. 상대 좌표로 환산
            rel_x = row[f"x{i}"] / 256
            rel_y = row[f"y{i}"] / 256
            # 2. 실제 이미지 해상도 기준 절대 좌표
            abs_x = int(rel_x * W)
            abs_y = int(rel_y * H)
            # 3. crop
            left = max(0, abs_x - self.crop_size // 2)
            top = max(0, abs_y - self.crop_size // 2)
            crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))

            if self.transform:
                crop = self.transform(crop)
            else:
                crop = torch.tensor(crop, dtype=torch.float32).unsqueeze(0) / 255.0

            crops.append(crop)

        crops_tensor = torch.stack(crops)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return crops_tensor, label
