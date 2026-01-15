import argparse
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import DefaultScope
from PIL import Image

from mmpretrain.apis import init_model
from mmpretrain.registry import TRANSFORMS


class SimpleCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


def build_model(config_path, checkpoint_path, device):
    cfg = Config.fromfile(config_path)
    model = init_model(cfg, checkpoint_path, device=device)
    model.eval()

    if hasattr(model, 'backbone') and hasattr(model.backbone, 'enable_sa_vis'):
        model.backbone.enable_sa_vis = True
    else:
        raise RuntimeError('Backbone does not support scale-aware visualization.')

    if hasattr(model, 'backbone') and hasattr(model.backbone, 'scale_aware'):
        if not model.backbone.scale_aware:
            raise RuntimeError(
                'The loaded backbone is not in SA-SSM (scale-aware) mode. '
                'Please use a config whose backbone.path_type contains "sa_ssm" '
                'and a matching checkpoint trained with that architecture.')

    return model, cfg


def build_transform(cfg):
    pipeline = []
    data_cfg = None
    if hasattr(cfg, 'val_dataloader'):
        data_cfg = cfg.val_dataloader.dataset
    elif hasattr(cfg, 'test_dataloader'):
        data_cfg = cfg.test_dataloader.dataset
    if data_cfg is not None:
        pipeline = data_cfg.pipeline

    import mmpretrain.datasets.transforms  # noqa: F401

    built = []
    with DefaultScope.overwrite_default_scope('mmpretrain'):
        for t_cfg in pipeline:
            built.append(TRANSFORMS.build(t_cfg))
    return SimpleCompose(built)


def tensor_to_heat_raw(feat_map, img_size, mode):
    if mode == 'mean':
        feat = feat_map.mean(dim=0, keepdim=True)
    elif mode == 'l2':
        feat = torch.sqrt((feat_map.float() ** 2).sum(dim=0, keepdim=True) + 1e-12)
    else:
        raise ValueError(f'Invalid heat mode: {mode}')

    feat = F.interpolate(
        feat.unsqueeze(0), size=img_size, mode='bilinear', align_corners=False
    ).squeeze(0).squeeze(0)
    return feat.cpu().numpy()


def normalize_heat(heat, vmin=None, vmax=None):
    if vmin is None:
        vmin = float(np.min(heat))
    if vmax is None:
        vmax = float(np.max(heat))
    return (heat - vmin) / (vmax - vmin + 1e-6)


def overlay(image, heat, alpha=0.5, cmap='jet'):
    cmap_func = plt.get_cmap(cmap)
    heat_color = cmap_func(heat)[..., :3]
    vis = alpha * heat_color + (1 - alpha) * image
    return np.clip(vis, 0, 1)


def load_base_image(img_path, img_size):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def prepare_inputs(img_path, transforms, model, device):
    data = dict(img_path=img_path)
    packed = transforms(data)
    img = packed['inputs'].unsqueeze(0).to(device)
    processed = model.data_preprocessor({'inputs': img}, training=False)
    return processed['inputs']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--images', type=str, nargs='+', required=True)
    parser.add_argument('--out-dir', type=str, default='outputs/sa_vis')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--layer-idx', type=int, default=-1)
    parser.add_argument('--heat-mode', type=str, default='mean', choices=['mean', 'l2'])
    parser.add_argument('--shared-norm', action='store_true')
    parser.add_argument('--save-diff', action='store_true')
    args = parser.parse_args()

    repo_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, cfg = build_model(args.config, args.checkpoint, device)

    if hasattr(model, 'backbone') and hasattr(model.backbone, 'sa_vis_layer_idx'):
        model.backbone.sa_vis_layer_idx = args.layer_idx

    transforms = build_transform(cfg)

    for img_path in args.images:
        inputs = prepare_inputs(img_path, transforms, model, device)

        with torch.no_grad():
            _ = model(inputs, mode='predict')

        if model.backbone.sa_vis_fine is None:
            raise RuntimeError(
                'No scale-aware visualization features were captured. '
                'Please check the config backbone.path_type and checkpoint compatibility.')

        fine = model.backbone.sa_vis_fine[0]
        coarse = model.backbone.sa_vis_coarse[0]
        fusion = model.backbone.sa_vis_fusion[0]

        img_size = (args.img_size, args.img_size)
        base_img = load_base_image(img_path, img_size=args.img_size)

        heat_fine_raw = tensor_to_heat_raw(fine, img_size, mode=args.heat_mode)
        heat_coarse_raw = tensor_to_heat_raw(coarse, img_size, mode=args.heat_mode)
        heat_fusion_raw = tensor_to_heat_raw(fusion, img_size, mode=args.heat_mode)

        if args.shared_norm:
            vmin = float(min(heat_fine_raw.min(), heat_coarse_raw.min(), heat_fusion_raw.min()))
            vmax = float(max(heat_fine_raw.max(), heat_coarse_raw.max(), heat_fusion_raw.max()))
            heat_fine = normalize_heat(heat_fine_raw, vmin=vmin, vmax=vmax)
            heat_coarse = normalize_heat(heat_coarse_raw, vmin=vmin, vmax=vmax)
            heat_fusion = normalize_heat(heat_fusion_raw, vmin=vmin, vmax=vmax)
        else:
            heat_fine = normalize_heat(heat_fine_raw)
            heat_coarse = normalize_heat(heat_coarse_raw)
            heat_fusion = normalize_heat(heat_fusion_raw)

        vis_fine = overlay(base_img, heat_fine)
        vis_coarse = overlay(base_img, heat_coarse)
        vis_fusion = overlay(base_img, heat_fusion)

        out_base = osp.splitext(osp.basename(img_path))[0]

        plt.imsave(osp.join(args.out_dir, f'{out_base}_input.png'), base_img)
        plt.imsave(osp.join(args.out_dir, f'{out_base}_fine.png'), vis_fine)
        plt.imsave(osp.join(args.out_dir, f'{out_base}_coarse.png'), vis_coarse)
        plt.imsave(osp.join(args.out_dir, f'{out_base}_fusion.png'), vis_fusion)

        if args.save_diff:
            diff = np.abs(heat_fusion - heat_fine)
            diff = normalize_heat(diff)
            vis_diff = overlay(base_img, diff, cmap='magma')
            plt.imsave(osp.join(args.out_dir, f'{out_base}_diff.png'), vis_diff)


if __name__ == '__main__':
    main()
