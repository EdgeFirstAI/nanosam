# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NanoSAM distillation training.

Pre-extract teacher features first with scripts/extract_sam_features.py,
then train the student encoder against those cached embeddings:

    python scripts/extract_sam_features.py \\
        --checkpoint mobile_sam.pt --model_type vit_t \\
        --img_dir /data/sa1b/images --out_dir /data/sa1b/features

    python nanosam/tools/train.py \\
        --images /data/sa1b/images \\
        --features /data/sa1b/features \\
        --output_dir runs/resnet18_distill
"""

import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, ToTensor, Normalize
import PIL.Image
from tqdm import tqdm

from nanosam.models.torch import create_model, list_models


# ---------------------------------------------------------------------------
# Dataset — pairs images with pre-extracted teacher .npy features
# ---------------------------------------------------------------------------

class FeaturePairDataset(Dataset):
    """Load image + pre-extracted teacher embedding pairs.

    Args:
        images_dir: Directory of .jpg / .png images.
        features_dir: Directory of .npy teacher embeddings with matching
            basenames (produced by scripts/extract_sam_features.py).
        img_size: Spatial size to resize images before feeding the student.
    """

    _EXTS = {".jpg", ".jpeg", ".png"}

    def __init__(self, images_dir: str, features_dir: str, img_size: int = 1024):
        self.img_size = img_size
        self.transform = Compose([
            ToTensor(),
            Normalize(
                mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
            ),
        ])

        # Collect image paths that have a matching .npy feature file
        all_images = []
        for ext in self._EXTS:
            all_images += glob.glob(os.path.join(images_dir, f"*{ext}"))

        self.pairs = []
        for img_path in sorted(all_images):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            feat_path = os.path.join(features_dir, stem + ".npy")
            if os.path.exists(feat_path):
                self.pairs.append((img_path, feat_path))

        if not self.pairs:
            raise FileNotFoundError(
                f"No matching image/feature pairs found.\n"
                f"  images_dir:   {images_dir}\n"
                f"  features_dir: {features_dir}\n"
                f"Run scripts/extract_sam_features.py first."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index):
        img_path, feat_path = self.pairs[index]
        image = PIL.Image.open(img_path).convert("RGB")
        if image.width != self.img_size or image.height != self.img_size:
            image = image.resize((self.img_size, self.img_size), PIL.Image.BILINEAR)
        image = self.transform(image)
        features = torch.from_numpy(np.load(feat_path).astype(np.float32)).squeeze(0)
        return image, features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NanoSAM distillation training")
    parser.add_argument(
        "--images", type=str, required=True,
        help="Directory of images used for distillation.",
    )
    parser.add_argument(
        "--features", type=str, required=True,
        help="Directory of pre-extracted teacher .npy embeddings "
             "(produced by scripts/extract_sam_features.py).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to store checkpoints and visualizations.",
    )
    parser.add_argument(
        "--model_name", type=str, default="resnet18", choices=list_models(),
        help="Student model name.",
    )
    parser.add_argument(
        "--student_size", type=int, default=1024,
        help="Spatial resolution fed to the student encoder.",
    )
    parser.add_argument(
        "--num_images", type=int, default=None,
        help="Limit images per epoch (useful for quick experiments).",
    )
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument(
        "--loss", type=str, default="huber", choices=["huber", "l1", "mse"],
        help="Distillation loss function.",
    )
    parser.add_argument("--log_step", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    # Student model
    student = create_model(args.model_name, args.student_size).cuda()

    loss_fn = {"huber": F.huber_loss, "l1": F.l1_loss, "mse": F.mse_loss}[args.loss]
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)

    dataset = FeaturePairDataset(args.images, args.features, img_size=args.student_size)
    print(f"Dataset: {len(dataset)} image/feature pairs")

    if args.num_images is not None:
        dataset, _ = random_split(dataset, [args.num_images, len(dataset) - args.num_images])

    loader = DataLoader(
        dataset, shuffle=True, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
    )

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.0
        prog_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch + 1}")

        for cnt, (image, features) in prog_bar:
            image = image.cuda(non_blocking=True)
            features = features.cuda(non_blocking=True)

            if len(image) != args.batch_size:
                continue

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = student(image)
                loss = loss_fn(output, features)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss)

            if (cnt + 1) % args.log_step == 0:
                prog_bar.set_postfix_str(f"loss: {epoch_loss / (cnt + 1):.5f}")

        epoch_loss /= len(loader)
        print(f"{epoch} - {epoch_loss:.5f}")

        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(f"{epoch} - {epoch_loss}\n")

        torch.save(
            {"model": student.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
            checkpoint_path,
        )

        # Visual sanity check — compare teacher vs student embedding channel 0
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(features[0, 0].detach().cpu().numpy())
        plt.title("Teacher")
        plt.subplot(122)
        plt.imshow(output[0, 0].detach().cpu().numpy())
        plt.title("Student")
        plt.suptitle(f"Epoch {epoch + 1}  loss={epoch_loss:.5f}")
        plt.savefig(os.path.join(args.output_dir, "images", f"epoch_{epoch}.png"))
        plt.close()
