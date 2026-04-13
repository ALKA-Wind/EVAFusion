import os
import argparse
import json
import importlib.util
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
from train import evaluate_model, HuggingFaceDataset


# Config
parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default=None, help="The path where trained model is saved.")
parser.add_argument("--ckpt", type=str, default=None, help="Or directly use the path to the trained model.")
parser.add_argument("--vit_model", type=str, default="vit-large-patch16-384", help="Name of the Vision Transformer model.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size in inference.")
parser.add_argument("--best", action="store_true", help="Whether to use the best model by val metrics.")
parser.add_argument("--infer", action="store_true", help="Do inference and visualization of heatmaps.")
parser.add_argument("--eval", action="store_true", help="Do evaluation and calculation of metrics.")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU device id (single GPU training)")
args = parser.parse_args()
print(args)


def import_model_from_path(script_path):
    """Load the model config from the log_dir"""  
    module_name = "rahf_model"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.RAHF

if args.log_dir is not None:
    # 从保存模型的python文件中加载模型
    RAHF = import_model_from_path(os.path.join(args.log_dir, 'model.py'))
    with open(os.path.join(args.log_dir, 'args_config.json')) as f:
        config = json.load(f)
else:
    from model import RAHF
    config = {'vit_model': 'vit-large-patch16-384', 'multi_heads': True}

vit_model = config['vit_model']

patch_size = 16
image_size = 384
if vit_model.startswith('google/vit'):
    patch_size = int(vit_model.split('-')[-2].replace('patch', ''))
    image_size = int(vit_model.split('-')[-1])

model = RAHF(
    vit_model=config['vit_model'], 
    multi_heads=config['multi_heads'], patch_size=patch_size, image_size=image_size
)

# 对模型进行推理，并保存推理结果
def infer(model, dataloader, device, log_dir, tag, max_iter=None):
    model.eval()

    data = defaultdict(list)

    iter_ = 0

    with torch.no_grad():
        # 循环多个batch，对数据进行处理
        for batch in tqdm(dataloader, leave=False):
            for key in list(batch.keys()):
                val = batch[key]
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)
        
            outputs = model(batch['fused'], batch['ir'], batch['vi'])

            # 收集预测值与真实值
            data['fused'].append(batch['fused'].cpu())

            # 预测评分
            out_scores = outputs['scores']
            for score in ('Thermal_Retention', 'Texture_Preservation', 'Artifacts', 'Sharpness', 'Overall_Score'):
                data[f'gt_score_{score}'].append(batch[score].cpu())
                data[f'pred_score_{score}'].append(out_scores[score].cpu())

            out_heatmaps = outputs['heatmaps']
            # for heatmap in ('artifact_heatmap', 'information_loss_heatmap'):
            for heatmap in ('artifact_heatmap', ):
                data[f'gt_heatmap_{heatmap}'].append(batch[heatmap].cpu())
                data[f'pred_heatmap_{heatmap}'].append(out_heatmaps[heatmap].cpu())

            for heatmap_name, heatmap_tensor in outputs['heatmaps'].items():
                if isinstance(heatmap_tensor, torch.Tensor):
                    heatmap = heatmap_tensor.squeeze().detach().cpu().numpy()
                    print(f"{heatmap_name} min: {heatmap.min():.4f}, max: {heatmap.max():.4f}")
                else:
                    print(f"Warning: {heatmap_name} is not a tensor, type: {type(heatmap_tensor)}")



            iter_ += 1
            if max_iter is not None and iter_ >= max_iter:
                break

    out_js = []

    # 将分数保存到JSON文件中输出
    for score in ('Thermal_Retention', 'Texture_Preservation', 'Artifacts', 'Sharpness', 'Overall_Score'):
        data[f'gt_score_{score}'] = torch.cat(data[f'gt_score_{score}']).flatten().tolist()
        data[f'pred_score_{score}'] = torch.cat(data[f'pred_score_{score}']).flatten().tolist()

    for i in range(len(data['fused'])):
        tmp = {'idx': i}
        for score in ('Thermal_Retention', 'Texture_Preservation', 'Artifacts', 'Sharpness', 'Overall_Score'):
            tmp[f'gt_score_{score}'] = data[f'gt_score_{score}'][i]
            tmp[f'pred_score_{score}'] = data[f'pred_score_{score}'][i]

        out_js.append(tmp)

    if args.best:
        tag += '_best'


    with open(os.path.join(log_dir, f'{tag}_outputs.json'), 'w') as f:
        json.dump(out_js, f, indent=2)

    # 图像归一化到0~1
    data['fused'] = torch.cat(data['fused']) * 0.5 + 0.5  # denormalize (works for google's ViTs)

    # for heatmap in ('artifact_heatmap', 'information_loss_heatmap'):
    for heatmap in ('artifact_heatmap', ):
        data[f'gt_heatmap_{heatmap}'] = torch.cat(data[f'gt_heatmap_{heatmap}']).unsqueeze(1).expand_as(data['fused'])
        data[f'pred_heatmap_{heatmap}'] = torch.cat(data[f'pred_heatmap_{heatmap}']).unsqueeze(1).expand_as(data['fused'])

    img_path = os.path.join(log_dir, f'{tag}_imgs')
    os.makedirs(img_path, exist_ok=True)

    # 拼接原图 GT 预测值，生成两排图片，第一排是GT，第二排是预测值
    for i in range(len(data['fused'])):
        imgs1 = [data['fused'][i]]
        imgs2 = [data['fused'][i]]
        # for heatmap in ('artifact_heatmap', 'information_loss_heatmap'):
        for heatmap in ('artifact_heatmap', ):
            imgs1.append(data[f'gt_heatmap_{heatmap}'][i])
            imgs2.append(data[f'pred_heatmap_{heatmap}'][i])
        imgs1 = torch.cat(imgs1, dim=2)
        imgs2 = torch.cat(imgs2, dim=2)
        imgs = torch.cat([imgs1, imgs2], dim=1)
        imgs = imgs.permute([1, 2, 0])  # change to h x w x 3
        im = Image.fromarray(imgs.byte().numpy())
        # imgs = imgs * 255
        # im = Image.fromarray(imgs.numpy().astype('uint8'))
        im.save(os.path.join(img_path, f'{i:03d}.png'))

# Load dataset
dataset_path = '/public/home/meiqingyun/tmp/IVIF/RichHF/data/test'
full_dataset = load_from_disk(dataset_path)


# Create PyTorch datasets and dataloaders
# train_dataset = HuggingFaceDataset(full_dataset["train"], vit_model, image_size)
# dev_dataset = HuggingFaceDataset(full_dataset["dev"], vit_model, image_size)
# test_dataset = HuggingFaceDataset(full_dataset["test"], vit_model, image_size)
test_dataset = HuggingFaceDataset(full_dataset["test"], vit_model, image_size)

# train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)#, collate_fn=collate_fn)
# dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)#, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)#, collate_fn=collate_fn)


device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
if args.log_dir is not None:
    ckpt_name = 'model_best.pt' if args.best else 'model.pt'
    ckpt = torch.load(os.path.join(args.log_dir, ckpt_name))
else:
    ckpt = torch.load(args.ckpt)
msg = model.load_state_dict(ckpt)
print("Loaded model:", msg)
model = model.to(device)


if args.eval:
    metrics = evaluate_model(model, test_dataloader, nn.MSELoss(), device, 'test')
    file = f"test_metrics{'_best' if args.best else ''}.json"
    with open(os.path.join(args.log_dir, file), 'wt') as f:
        json.dump(metrics, f, indent=4)

if args.infer:
    infer(model, test_dataloader, device, args.log_dir, 'test')
    # infer(model, test_dataloader, device, args.log_dir, 'test')
    # infer(model, dev_dataloader, device, args.log_dir, 'dev')
    # infer(model, train_dataloader, device, args.log_dir, 'train', max_iter=10)