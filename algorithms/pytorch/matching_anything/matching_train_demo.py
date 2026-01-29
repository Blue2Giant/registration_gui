#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile, UnidentifiedImageError
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve()))
sys.path.append(str((Path(__file__).parent / "third_party" / "ROMA").resolve()))

from third_party.ROMA.roma.matchanything_roma_model import MatchAnything_Model


class SAROPTTrainDataset(Dataset):
    def __init__(
        self,
        txt_file,
        replace_from="/SAR/",
        replace_to="/OPT/",
        size=512,
        color_mode="RGB",
        skip_missing=True,
        resize_by_stretch=True,
    ) -> None:
        super().__init__()
        self.replace_from = replace_from
        self.replace_to = replace_to
        self.color_mode = color_mode
        self.skip_missing = skip_missing
        self.resize_by_stretch = resize_by_stretch

        with open(txt_file, "r", encoding="utf-8") as f:
            sar_list = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
        self.pairs = []
        for sar in sar_list:
            opt = sar.replace(self.replace_from, self.replace_to)
            if os.path.exists(sar) and os.path.exists(opt):
                self.pairs.append((sar, opt))
        if len(self.pairs) == 0:
            raise RuntimeError("No valid SAR/OPT pairs.")

        self.to_tensor = transforms.ToTensor()
        self.resize_size = size

    def __len__(self):
        return len(self.pairs)

    def _load(self, p):
        try:
            im = Image.open(p)
            im = im.convert(self.color_mode)
            return im
        except (UnidentifiedImageError, OSError):
            if self.skip_missing:
                return None
            raise

    def _resize_pad_or_stretch(self, t):
        c, h, w = t.shape
        if self.resize_by_stretch:
            im = transforms.Resize((self.resize_size, self.resize_size), interpolation=transforms.InterpolationMode.BICUBIC)(t)
            return im, torch.tensor([self.resize_size, self.resize_size], dtype=torch.long)
        scale = self.resize_size / max(h, w)
        hs, ws = round(h * scale), round(w * scale)
        im = transforms.Resize((hs, ws), interpolation=transforms.InterpolationMode.BICUBIC)(t)
        out = torch.zeros((c, self.resize_size, self.resize_size), dtype=im.dtype)
        out[:, :hs, :ws] = im
        return out, torch.tensor([hs, ws], dtype=torch.long)

    def __getitem__(self, idx):
        sar_p, opt_p = self.pairs[idx]
        sar = self._load(sar_p)
        opt = self._load(opt_p)
        if sar is None or opt is None:
            return {"_invalid": True}
        sar_t = self.to_tensor(sar)
        opt_t = self.to_tensor(opt)
        im0, sz0 = self._resize_pad_or_stretch(sar_t)
        im1, sz1 = self._resize_pad_or_stretch(opt_t)
        return {
            "image0": im0,
            "image1": im1,
            "origin_img_size0": sz0,
            "origin_img_size1": sz1,
        }

    @staticmethod
    def collate_fn(batch):
        valid = [b for b in batch if not b.get("_invalid")]
        if len(valid) == 0:
            return {
                "image0": torch.empty(0),
                "image1": torch.empty(0),
                "origin_img_size0": torch.empty(0, 2, dtype=torch.long),
                "origin_img_size1": torch.empty(0, 2, dtype=torch.long),
            }
        image0 = torch.stack([b["image0"] for b in valid], 0)
        image1 = torch.stack([b["image1"] for b in valid], 0)
        size0 = torch.stack([b["origin_img_size0"] for b in valid], 0)
        size1 = torch.stack([b["origin_img_size1"] for b in valid], 0)
        return {
            "image0": image0,
            "image1": image1,
            "origin_img_size0": size0,
            "origin_img_size1": size1,
        }


def tv_loss(flow):
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def build_model(args):
    cfg = {
        "resize_by_stretch": args.resize_by_stretch,
        "normalize_img": False,
        "model": {
            "amp": args.amp,
            "coarse_backbone": "DINOv2_large",
            "coarse_feat_dim": 1024,
            "medium_feat_dim": 512,
            "coarse_patch_size": 14,
        },
        "test_time": {
            "coarse_res": 512,
            "symmetric": False,
            "upsample": False,
            "attenutate_cert": False,
            "upsample_res": 512,
        },
        "sample": {"method": "bilinear", "thresh": 0.0, "n_sample": 4096},
        "match_thresh": 0.3,
    }
    model = MatchAnything_Model(cfg, test_mode=False)
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--txt_file", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps_per_epoch", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resize_by_stretch", action="store_true")
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--ckpt_out", type=str, default="roma_sar_opt.pth")
    return p.parse_args()


def main():
    args = parse_args()
    ds = SAROPTTrainDataset(
        txt_file=args.txt_file,
        size=args.size,
        resize_by_stretch=args.resize_by_stretch,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=SAROPTTrainDataset.collate_fn,
        drop_last=True,
    )
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = build_model(args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    step = 0
    for epoch in range(args.epochs):
        it = iter(dl)
        for _ in range(args.steps_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            image0 = batch["image0"].to(device)
            image1 = batch["image1"].to(device)
            origin0 = batch["origin_img_size0"].to(device)
            origin1 = batch["origin_img_size1"].to(device)
            data = {
                "image0": image0,
                "image1": image1,
                "origin_img_size0": origin0,
                "origin_img_size1": origin1,
            }
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=args.amp):
                model.forward_train_framework(data)
                corresps = data["corresps"][1]
                flow = corresps["flow"]
                cert_logits = corresps["certainty"]
                loss_cert = -torch.sigmoid(cert_logits).mean()
                loss_smooth = tv_loss(flow)
                loss = loss_cert + 0.1 * loss_smooth
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if args.save_every and step % args.save_every == 0:
                torch.save({"model": model.state_dict(), "step": step}, args.ckpt_out)
        if not args.save_every:
            torch.save({"model": model.state_dict(), "step": step}, args.ckpt_out)


if __name__ == "__main__":
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    main()
