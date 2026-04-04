import io
import os
import shutil
import datetime
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torchvision import transforms
from transformers import AutoImageProcessor
from PIL import Image
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from model import RAHF
from metrics import metrics_func, mse_loss
from torch.utils.tensorboard import SummaryWriter


class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, vit_model="vit-large-patch16-384", image_size=384):
        self.dataset = hf_dataset
        self.image_transform = AutoImageProcessor.from_pretrained(vit_model)
        self.heatmap_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def preprocess_heatmap(self, image):
        if image.mode != 'L':
            image = image.convert('L')
        return self.heatmap_transform(image).squeeze(0)
    
    
    
    def __getitem__(self, idx):
        score_max = 5.0
        sample = self.dataset[idx]
        out = {}
        fused = sample["fused"]
        ir = sample["infrared"]
        vi = sample["visible"]
        ir = to_rgb(sample["infrared"])
        vi = to_rgb(sample["visible"])
        fused = to_rgb(sample["fused"])
        out['fused'] = self.image_transform(fused, return_tensors="pt")['pixel_values'][0]
        out['ir'] = self.image_transform(ir, return_tensors="pt")['pixel_values'][0]
        out['vi'] = self.image_transform(vi, return_tensors="pt")['pixel_values'][0]
        # img = Image.open(io.BytesIO(sample["image"]))
        # out['image'] = self.image_transform(img, return_tensors="pt")['pixel_values'][0]

        out['artifact_heatmap'] = self.preprocess_heatmap(sample["artifact_heatmap"])
        # out['information_loss_heatmap'] = self.preprocess_heatmap(sample["information_loss_heatmap"])

        scores = sample["scores"]
        out['Thermal_Retention'] = np.array(scores['Thermal Retention'] / score_max, dtype=np.float32) 
        out['Texture_Preservation'] = np.array(scores['Texture Preservation'] / score_max, dtype=np.float32)
        out['Artifacts'] = np.array(scores['Artifacts'] / score_max, dtype=np.float32)
        out['Sharpness'] = np.array(scores['Sharpness'] / score_max, dtype=np.float32)
        out['Overall_Score'] = np.array(scores['Overall Score'] / score_max, dtype=np.float32)
        
        return out


def to_rgb(image):
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                return image.convert('RGB')
            return image
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:  # (H, W)
                image = np.stack([image] * 3, axis=-1)  # -> (H, W, 3)
            elif image.ndim == 3 and image.shape[2] == 1:  # (H, W, 1)
                image = np.repeat(image, 3, axis=2)
            return Image.fromarray(image)
        else:
            raise TypeError("Unsupported image type")

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    target_images = torch.stack([item["target_image"] for item in batch])
    captions = [item["caption"] for item in batch]
    target_texts = [item["target_text"] for item in batch]
    scores = torch.stack([item["score"] for item in batch])
    return {"images": images, "target_images": target_images, "captions": captions, "target_texts": target_texts, "scores": scores}


# Training function
def train_model(model, dataloader, optimizer, mse_criterion, device):
    model.train()
    total_loss = []
    iter_ = 0
    loss_all = defaultdict(list)

    # 数据移动到计算设备
    for batch in tqdm(dataloader, desc="Training", leave=False):
        for key in list(batch.keys()):
            val = batch[key]
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(device)

        # 上一批次梯度清零
        optimizer.zero_grad()

        # 将图像传入模型，得到预测出的热图和分数
        outputs = model(batch['fused'], batch['ir'], batch['vi'])

        # 计算真实值与预测值差异，计算损失
        loss_dict = {}
        total_batch_loss = 0
        out_scores = outputs['scores']
        for score in ('Thermal_Retention', 'Texture_Preservation', 'Artifacts', 'Sharpness', 'Overall_Score'):
            loss_dict[score] = mse_criterion(batch[score], out_scores[score])
            total_batch_loss += loss_dict[score] * args.score_weight

        out_heatmaps = outputs['heatmaps']
        # for heatmap in ('artifact_heatmap', 'information_loss_heatmap'):
        for heatmap in ('artifact_heatmap',):
            loss_dict[heatmap] = mse_criterion(batch[heatmap], out_heatmaps[heatmap])
            total_batch_loss += loss_dict[heatmap] * args.heatmap_weight

        # 反向传播与参数更新​
        total_batch_loss.backward()
        optimizer.step()

        # 记录损失
        for key, val in loss_dict.items():
            loss_all[key].append(val.item())

        total_loss.append(total_batch_loss.item())
        iter_ += 1

    # 计算平均损失
    for key in list(loss_all.keys()):
        loss_all[key] = np.mean(loss_all[key])

    return np.mean(total_loss), loss_all


def avg_metric(metrics):
    """Compute an overall metric for the rich feedback metrics"""
    s = 0
    for sc in ['Thermal_Retention', 'Texture_Preservation', 'Artifacts', 'Sharpness', 'Overall_Score']:
        for mt in ['plcc', 'srcc']:
            s += metrics[f"score/{mt}/{sc}"]
    
    h = 0
    # for hm in ['artifact_heatmap', 'information_loss_heatmap']:
    for hm in ['artifact_heatmap']:
        for mt in ['nss', 'cc', 'similarity', 'AUC_Judd']:
            h += metrics[f"heatmap/{mt}/{hm}"]


    avg = s / 10 + + h / 4
    return avg


# def evaluate_model(model, dataloader, mse_criterion, device, phase="Validation"):
#     model.eval()

#     predicted_scores = defaultdict(list)
#     target_scores = defaultdict(list)
#     loss_dict = defaultdict(list)
#     total_batch_loss = 0.0

#     heatmap_metrics = defaultdict(list)

#     gt_texts = []
#     pred_texts = []

#     # 数据迁移到设备
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc=f"{phase}", leave=False):
#             for key in list(batch.keys()):
#                 val = batch[key]
#                 if isinstance(val, torch.Tensor):
#                     batch[key] = val.to(device)

#             outputs = model(batch['fused'], batch['ir'], batch['vi'])

#             # 损失计算
#             out_scores = outputs['scores']
#             for score in ('Thermal_Retention', 'Texture_Preservation', 'Artifacts', 'Sharpness', 'Overall_Score'):
#                 loss_dict[score].append(mse_criterion(batch[score], out_scores[score]).item())
#                 total_batch_loss += loss_dict[score][-1]  * args.score_weight
            
#                 predicted_scores[score].append(out_scores[score].cpu().numpy())
#                 target_scores[score].append(batch[score].cpu().numpy())

#             out_heatmaps = outputs['heatmaps']
#             # for heatmap in ('artifact_heatmap', 'information_loss_heatmap'):
#             for heatmap in ('artifact_heatmap',):
#                 gt = batch[heatmap]
#                 pred = out_heatmaps[heatmap]
#                 loss_dict[heatmap].append(mse_criterion(gt, pred).item())
#                 total_batch_loss += loss_dict[heatmap][-1] * args.heatmap_weight

#                 gt_sum = gt.sum(dim=[1, 2])
#                 nonzero_gt = gt_sum > 0
#                 if torch.any(~nonzero_gt):
#                     mse_l = mse_loss(pred[~nonzero_gt], gt[~nonzero_gt])
#                     heatmap_metrics[f'mse/{heatmap}'].append(mse_l)
#                 if torch.any(nonzero_gt):
#                     pred_, gt_ = pred[nonzero_gt], gt[nonzero_gt]
#                     for key, func in metrics_func.items():
#                         metric_val = func(pred_, gt_)
#                         heatmap_metrics[f'{key}/{heatmap}'].append(metric_val)

#             loss_dict['total_loss'].append(total_batch_loss.item())

#     # 指标汇总
#     metrics = {}
#     for score in ('Thermal_Retention', 'Texture_Preservation', 'Artifacts', 'Sharpness', 'Overall_Score'):
#         # 合并预测值和真实值
#         pred = np.concatenate(predicted_scores[score])
#         gt = np.concatenate(target_scores[score])
#         metrics[f'score/plcc/{score}'] = pearsonr(pred, gt)[0]
#         metrics[f'score/srcc/{score}'] = spearmanr(pred, gt)[0]
#         metrics[f'loss/{score}'] = np.mean(loss_dict[score])

#     # for heatmap in ('artifact_heatmap', 'information_loss_heatmap'):
#     for heatmap in ('artifact_heatmap',):
#         metrics[f'loss/{heatmap}'] = np.mean(loss_dict[heatmap])

#     for key, vals in heatmap_metrics.items():
#         val = torch.cat(vals).mean().item()
#         metrics[f"heatmap/{key}"] = val

#     metrics['loss/val'] = np.mean(loss_dict['total_loss'])

#     metrics['avg_metric'] = avg_metric(metrics)
#     return metrics

def evaluate_model(model, dataloader, mse_criterion, device, phase="Validation"):
    model.eval()

    predicted_scores = defaultdict(list)
    target_scores = defaultdict(list)
    loss_dict = defaultdict(list)
    heatmap_metrics = defaultdict(list)
    total_batch_loss_list = []  # ✅ 用来记录每个 batch 的 loss

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{phase}", leave=False):
            for key in list(batch.keys()):
                val = batch[key]
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)

            outputs = model(batch['fused'], batch['ir'], batch['vi'])
            total_batch_loss = 0.0  # ✅ 每个 batch 初始化

            # Score loss
            out_scores = outputs['scores']
            for score in ('Thermal_Retention', 'Texture_Preservation', 'Artifacts', 'Sharpness', 'Overall_Score'):
                loss_val = mse_criterion(batch[score], out_scores[score]).item()
                loss_dict[score].append(loss_val)
                total_batch_loss += loss_val * args.score_weight

                predicted_scores[score].append(out_scores[score].cpu().numpy())
                target_scores[score].append(batch[score].cpu().numpy())

            # Heatmap loss
            out_heatmaps = outputs['heatmaps']
            for heatmap in ('artifact_heatmap',):
                gt = batch[heatmap]
                pred = out_heatmaps[heatmap]
                heatmap_loss = mse_criterion(gt, pred).item()
                loss_dict[heatmap].append(heatmap_loss)
                total_batch_loss += heatmap_loss * args.heatmap_weight

                gt_sum = gt.sum(dim=[1, 2])
                nonzero_gt = gt_sum > 0
                if torch.any(~nonzero_gt):
                    mse_l = mse_loss(pred[~nonzero_gt], gt[~nonzero_gt])
                    heatmap_metrics[f'mse/{heatmap}'].append(mse_l)
                if torch.any(nonzero_gt):
                    pred_, gt_ = pred[nonzero_gt], gt[nonzero_gt]
                    for key, func in metrics_func.items():
                        metric_val = func(pred_, gt_)
                        heatmap_metrics[f'{key}/{heatmap}'].append(metric_val)

            total_batch_loss_list.append(total_batch_loss)  # ✅ 直接 append float，无需 .item()

    # 汇总指标
    metrics = {}
    for score in ('Thermal_Retention', 'Texture_Preservation', 'Artifacts', 'Sharpness', 'Overall_Score'):
        pred = np.concatenate(predicted_scores[score])
        gt = np.concatenate(target_scores[score])
        metrics[f'score/plcc/{score}'] = pearsonr(pred, gt)[0]
        metrics[f'score/srcc/{score}'] = spearmanr(pred, gt)[0]
        metrics[f'loss/{score}'] = np.mean(loss_dict[score])

    for heatmap in ('artifact_heatmap',):
        metrics[f'loss/{heatmap}'] = np.mean(loss_dict[heatmap])

    for key, vals in heatmap_metrics.items():
        val = torch.cat(vals).mean().item()
        metrics[f"heatmap/{key}"] = val

    metrics['loss/val'] = np.mean(total_batch_loss_list)
    metrics['avg_metric'] = avg_metric(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training and evaluation for RAHF.")

    parser.add_argument("--vit_model", type=str, default="vit-large-patch16-384", help="Name of the Vision Transformer model.")
    parser.add_argument("--multi_heads", action="store_true", help="Whether to use multi-heads version or the augmented prompt version.")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--init_lr", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate.")
    parser.add_argument("--decoder_lr_scale", type=float, default=1, help="Scale the decoder LR.")
    parser.add_argument("--score_weight", type=float, default=1, help="Loss weight for score loss.")
    parser.add_argument("--heatmap_weight", type=float, default=1, help="Loss weight for heatmap loss.")

    # 选择合适的batch_size
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--freeze_vit", action="store_true", help="Freeze the ViT weights.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to the checkpoint to resume.")
    parser.add_argument("--log_dir", type=str, default="exp", help="Path (with an additional time stamp) to save model checkpoints and logs.")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training (for DDP).")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device id (single GPU training)")

    args = parser.parse_args()
    print(args)

    # Distributed Setup
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        distributed = True
        # Use the environment variable LOCAL_RANK if available.
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        if local_rank == 0:
            print("Distributed training with {} GPUs".format(os.environ["WORLD_SIZE"]))
    else:
        distributed = False
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    # Model and Dataset Setup
    vit_model = args.vit_model

    patch_size = 16
    image_size = 384
    if vit_model.startswith('google/vit'):
        patch_size = int(vit_model.split('-')[-2].replace('patch', ''))
        image_size = int(vit_model.split('-')[-1])
    elif vit_model == 'facebook/dino-vitb8':
        patch_size = 8
        image_size = 224
    elif vit_model == 'facebook/dino-vitb16':
        patch_size = 16
        image_size = 224

    # Load dataset
    train_path = '/public/home/meiqingyun/tmp/IVIF/RichHF/data/train'
    dev_path = '/public/home/meiqingyun/tmp/IVIF/RichHF/data/dev'
    full_traindataset = load_from_disk(train_path)
    full_devdataset = load_from_disk(dev_path)

    # Create PyTorch datasets
    train_dataset = HuggingFaceDataset(full_traindataset["train"], args.vit_model, image_size)
    dev_dataset = HuggingFaceDataset(full_devdataset["dev"], args.vit_model, image_size)
    # test_dataset = HuggingFaceDataset(full_dataset["test"], args.vit_model, image_size)


    # missing_key_indices = []
    # zero_heatmap_indices = []

    # print("Checking artifact_heatmap presence and values for all samples...")
    # for i in range(int(len(dev_dataset))):
    #     try:
    #         sample = dev_dataset[i]
    #         if "artifact_heatmap" not in sample:
    #             missing_key_indices.append(i)
    #         elif sample["artifact_heatmap"].sum() == 0:
    #             zero_heatmap_indices.append(i)
    #     except Exception as e:
    #         print(f"Error processing sample {i}: {e}")
    #         missing_key_indices.append(i)

    # print(f"\nTotal samples: {int(len(dev_dataset))}")
    # print(f"Samples missing 'artifact_heatmap' key: {len(missing_key_indices)}")
    # print(f"Samples with all-zero 'artifact_heatmap': {len(zero_heatmap_indices)}")

    # print("\nSample indices with missing key:", missing_key_indices)
    # print("Sample indices with zero-only heatmap:", zero_heatmap_indices)


    # Create dataloaders
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
        # Only rank 0 will run evaluation and logging
        if local_rank == 0:
            dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        #     test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        else:
            dev_dataloader = None
        #     test_dataloader = None
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model, optimizer, and loss functions
    model = RAHF(
        vit_model=args.vit_model,  
        multi_heads=args.multi_heads, patch_size=patch_size, image_size=image_size
    )
    if args.ckpt:
        msg = model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
        print(f"Loaded from ckpt:", msg)
    model = model.to(device)

    for n, p in model.named_parameters():
        if 'vit.pooler.dense' in n:  # not used in RAHF
            p.requires_grad = False

    # Wrap model in DDP if distributed
    if distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)

    raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    # 冻结两个 ViT 模块的参数
    for name, param in raw_model.vit.named_parameters():
        param.requires_grad = False
    for name, param in raw_model.vit_fusion.named_parameters():
        param.requires_grad = False

    # 计算可训练参数量
    num_parameters = 0
    p_other = []
    for n, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if local_rank == 0:
            print("可训练参数为：" + n)
        p_other.append(p)
        num_parameters += p.numel()

    if local_rank == 0:
        print("number of trainable parameters: %d" % num_parameters)
    # Freeze VIT 冻结两个VIT的参数
    # for name, param in model.named_parameters():
    #     for name, param in model.vit.named_parameters():
    #         param.requires_grad = False
    #     for name, param in model.vit_fusion.named_parameters():
    #         param.requires_grad = False

    # # raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    # # # === 冻结 ViT 模块（主模型和融合模型） ===
    # # for name, param in raw_model.vit.named_parameters():
    # #     param.requires_grad = False
    # # for name, param in raw_model.vit_fusion.named_parameters():
    # #     param.requires_grad = False
    # # 计算可训练参数量
    # num_parameters = 0
    # p_other =[]
    # # n为参数名称，p为参数值
    # for n, p in model.named_parameters():
    #     if not p.requires_grad:
    #         continue  # frozen weights
    #     # 输出可训练参数的名称
    #     if local_rank == 0:
    #         print("可训练参数为："+n)
    #     p_other.append(p)
    #     num_parameters += p.data.nelement()
    # if local_rank == 0:
    #     print("number of trainable parameters: %d" % num_parameters)
    

    
    optim_params = [
        {"params": p_other, "lr": args.init_lr},
    ]

    optimizer = optim.AdamW(optim_params, lr=args.init_lr, weight_decay=2e-3)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, args.min_lr)
    mse_criterion = nn.MSELoss()

    # Setup logging (only on rank 0)
    if local_rank == 0:
        tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(args.log_dir, tag)
        writer = SummaryWriter(log_dir=log_dir)
        print("log_dir:", log_dir)

        # Save the scripts
        script_path = os.path.abspath(__file__)
        shutil.copy(script_path, os.path.join(log_dir, 'train.py'))
        dirname = os.path.dirname(script_path)
        shutil.copy(os.path.join(dirname, 'model.py'), os.path.join(log_dir, 'model.py'))

        # Save the arguments config
        args_dict = vars(args)
        args_dict['log_dir'] = log_dir
        with open(os.path.join(log_dir, "args_config.json"), "w") as json_file:
            json.dump(args_dict, json_file, indent=2)
    else:
        log_dir = None

    # Training loop
    val_metrics_ = []
    max_val_metric = 0
    train_loss_ = []

    for epoch in range(args.num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        # 打印当前epoch，确保只有第一个进程打印信息，避免重复输出
        if local_rank == 0:
            print(f"Epoch {epoch + 1}/{args.num_epochs}")

        # Training phase
        train_loss, loss_all = train_model(model, train_dataloader, optimizer, mse_criterion, device)
        if local_rank == 0:
            print(f"Training Loss: {train_loss:.4f}")
            writer.add_scalar('loss/train', train_loss, epoch)
            for key, val in loss_all.items():
                writer.add_scalar(f'loss/train/{key}', val, epoch)
            train_loss_.append(loss_all)
            # 保存损失到JSON文件
            with open(os.path.join(log_dir, "train_loss.json"), 'wt') as f:
                json.dump(train_loss_, f, indent=4)

            # Validation phase
            val_metrics = evaluate_model(model, dev_dataloader, mse_criterion, device, phase="Validation")
            print(f"Validation Metrics: {val_metrics}")
            for m, v in val_metrics.items():
                writer.add_scalar(m, v, epoch)
            
            if val_metrics['avg_metric'] > max_val_metric:
                max_val_metric = val_metrics['avg_metric']
                print("Save best checkpoint at epoch", epoch)
                if distributed:
                    torch.save(model.module.state_dict(), os.path.join(log_dir, "model_best.pt"))
                else:
                    torch.save(model.state_dict(), os.path.join(log_dir, "model_best.pt"))

            lr_scheduler.step(epoch)
            lrs = lr_scheduler.get_lr()
            writer.add_scalar("LR", lrs[0], epoch)

            # 保存验证指标到JSON文件
            val_metrics['epoch'] = epoch
            val_metrics_.append(val_metrics)
            with open(os.path.join(log_dir, "val_metrics.json"), 'wt') as f:
                json.dump(val_metrics_, f, indent=4)

            if distributed:
                torch.save(model.module.state_dict(), os.path.join(log_dir, "model.pt"))
            else:
                torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))
        else:
            # For non-zero ranks, you may want to step the LR scheduler as well.
            lr_scheduler.step(epoch)

    # Testing phase (only on rank 0)
    # if local_rank == 0:
    #     test_metrics = evaluate_model(model, test_dataloader, mse_criterion, device, phase="Testing")
    #     print(f"Test Metrics: {test_metrics}")
    #     with open(os.path.join(log_dir, "test_metrics.json"), 'wt') as f:
    #         json.dump(test_metrics, f, indent=4)

    if distributed:
        torch.distributed.destroy_process_group()