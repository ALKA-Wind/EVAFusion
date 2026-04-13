import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from utils.loss import Fusionloss, cc
import copy
# Import DCEvo components instead of CDDFuse
from sleepnet import DE_Encoder, DE_Decoder, LowFreqExtractor, HighFreqExtractor
from utils.dataset import H5Dataset
# from utils.Evaluator import Evaluator
from reward_model.model import RAHF
from kornia.losses.ssim import SSIM
import kornia
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import sys
import time
import datetime
import torch.optim as optim
from torch.autograd import Variable
import logging
from mask import map_generate2
import cv2
import numpy as np 
from pseudo_fuse import pseudo_fuse_cuda
from torch.cuda.amp import autocast, GradScaler

# -------------- 配置 --------------
lr = 1e-4
weight_decay = 0
# batch_size = 2
batch_size = 1
num_epochs = 10
β_kl = 0.1

# Fusionloss 与各 reward 项权重
w_artifact = 0.5
w_texture = 2.0
w_thermal = 1.0
w_sharp = 2.0

# 伪融合损失和分解损失权重
fusion_loss_weight = 1.0
l1_loss_weight = 1.0
pseudo_fusion_weight = 100.0
decomp_loss_weight = 2.0
reward_loss_weight = 0.5

# scheduler 配置
step_size = 10
gamma = 0.5
use_cosine = False
# ----------------------------------

# 日志
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(log_dir, f"reward_new_fusionloss_train_log_{timestamp}.log")

logging.basicConfig(
    filename=log_file,
    filemode='w',
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
logging.info("Training started.")

# 纯奖励分数反馈，使用DCEvo架构
class FusionModel(nn.Module):
    def __init__(self, encoder, decoder, low_freq_extractor, high_freq_extractor):
        super(FusionModel, self).__init__()
        self.Encoder = encoder
        self.Decoder = decoder
        self.LowFreqExtractor = low_freq_extractor
        self.HighFreqExtractor = high_freq_extractor

    def forward(self, ir, vi):
        feature_V_B, feature_V_D, _ = self.Encoder(vi)
        feature_I_B, feature_I_D, _ = self.Encoder(ir)

        feature_F_B = self.LowFreqExtractor(feature_V_B + feature_I_B)
        feature_F_D = self.HighFreqExtractor(feature_V_D + feature_I_D)

        # DCEvo在解码时会使用输入的平均值
        fused, _ = self.Decoder(vi*0.5 + ir*0.5, feature_F_B, feature_F_D)
        return fused

def resize_to_384(img):
    return F.interpolate(img, size=(384, 384), mode='bilinear', align_corners=False)

def compute_white_ratio_batch(heatmaps, threshold=0.5):
    """
    输入:
      heatmaps: dict of Tensor, 每个值为 [B, 1, H, W] 或 [B, H, W]，值在 [0,1] 之间
      threshold: float，判定为"白色"的阈值
    返回:
      white_ratio: Tensor of shape [B,]，每个样本白色像素占比
    """
    ratios = []
    for name, hm in heatmaps.items():
        # hm -> [B, H, W]
        if hm.dim() == 4:
            hm = hm[:, 0]
        # 在 GPU 上做二值化
        mask = (hm >= threshold).to(torch.float32)  # [B, H, W]
        # 计算每张图的白色像素数 / (H*W)
        # sum over H,W -> [B,]
        white_pixels = mask.flatten(1).sum(dim=1)
        total_pixels = mask.shape[1] * mask.shape[2]
        ratios.append(white_pixels / total_pixels)  # [B,]
    # 如果有多个 heatmap，取它们的平均（或加权），这里示例直接平均
    white_ratio = torch.stack(ratios, dim=0).mean(dim=0)  # [B,]
    return white_ratio

# 融合损失
criteria_fusion = Fusionloss()
Loss_ssim = kornia.losses.SSIM(11, reduction='mean')

if __name__ == "__main__":

    # 训练优化参数
    optim_step = 20
    optim_gamma = 0.5
    # coeff_decomp = 2
    coeff_decomp = decomp_loss_weight
    # 混合精度
    scaler = GradScaler()

    # Determine the device (GPU if available, otherwise CPU) from the model's parameters.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载融合网络和奖励模型
    # 修改为DCEvo模型路径
    model_path = "ckpt/DCEvo_fusion.pth"
    reward_Model_path = "reward_model/20250503_232931/model_best.pt"
    reward_Model = RAHF().to(device)
    print("Loading model structure...")

    # 数据并行 加载到GPU - 使用DCEvo的组件
    Encoder = nn.DataParallel(DE_Encoder()).to(device)
    Decoder = nn.DataParallel(DE_Decoder()).to(device)
    LowFreqExtractor = nn.DataParallel(LowFreqExtractor(dim=64)).to(device)
    HighFreqExtractor = nn.DataParallel(HighFreqExtractor(num_layers=3)).to(device)

    # Load pre-trained model weights - 更新为DCEvo的键名
    Encoder.load_state_dict(torch.load(model_path)['DE_Encoder'])
    Decoder.load_state_dict(torch.load(model_path)['DE_Decoder'])
    LowFreqExtractor.load_state_dict(torch.load(model_path)['LowFreqExtractor'])
    HighFreqExtractor.load_state_dict(torch.load(model_path)['HighFreqExtractor'])
    reward_Model.load_state_dict(torch.load(reward_Model_path))

    # 把融合网络打包到一起
    model = FusionModel(Encoder, Decoder, LowFreqExtractor, HighFreqExtractor)
    # 进行并行处理
    model = nn.DataParallel(model).cuda()  

    optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, weight_decay=weight_decay) 
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_step, gamma=optim_gamma)
    T_max = num_epochs  # 也可以设为更大，例如 num_epochs * len(trainloader) if 按 iteration 调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)


    # trainloader = DataLoader(H5Dataset(r"/public/home/meiqingyun/tmp/IVIF/MMIF-CDDFuse/MSRS_patches_384.h5"),
    trainloader = DataLoader(H5Dataset("/public/home/meiqingyun/tmp/IVIF/MMIF-CDDFuse/MSRS_patches_128.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=6)

    step = 0
    prev_time = time.time()

    MSELoss = nn.MSELoss()  
    torch.backends.cudnn.benchmark = True

    # 冻结 reward_Model 的参数（不会更新权重），但可以反向传播
    for param in reward_Model.parameters():
        param.requires_grad = False

    for epoch in range(num_epochs):
        ''' Fine-tune '''
        # for i, (data_VIS, data_IR) in enumerate(trainloader):
        for i, (data_VIS, data_IR, data_masks) in enumerate(trainloader):
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # with autocast():
            if 1:
                # 使用更新后的网络进行图像融合 - 现在使用DCEvo架构
                new_feature_V_B, new_feature_V_D, _ = model.module.Encoder(data_VIS)
                new_feature_I_B, new_feature_I_D, _ = model.module.Encoder(data_IR)

                new_feature_F_B = model.module.LowFreqExtractor(new_feature_I_B+new_feature_V_B)
                new_feature_F_D = model.module.HighFreqExtractor(new_feature_I_D+new_feature_V_D)

                # DCEvo在解码时使用输入图像的平均值
                new_data_Fuse, _ = model.module.Decoder(data_VIS*0.5 + data_IR*0.5, new_feature_F_B, new_feature_F_D)

                # 计算融合损失
                # mse_loss_V = 5*Loss_ssim(data_VIS, new_data_Fuse) + MSELoss(data_VIS, new_data_Fuse)
                # mse_loss_I = 5*Loss_ssim(data_IR,  new_data_Fuse) + MSELoss(data_IR,  new_data_Fuse)
                mse_loss_V = 5*Loss_ssim(data_VIS, new_data_Fuse) + l1_loss_weight * MSELoss(data_VIS, new_data_Fuse)
                mse_loss_I = 5*Loss_ssim(data_IR,  new_data_Fuse) + l1_loss_weight * MSELoss(data_IR,  new_data_Fuse)

                cc_loss_B = cc(new_feature_V_B, new_feature_I_B)
                cc_loss_D = cc(new_feature_V_D, new_feature_I_D)
                loss_decomp =   (cc_loss_D) ** 2 / (1.01 + cc_loss_B)  
                fusionloss, _,_  = criteria_fusion(data_VIS, data_IR, new_data_Fuse)

                # loss1 = fusionloss + coeff_decomp * loss_decomp
                loss1 = fusion_loss_weight * fusionloss + decomp_loss_weight * loss_decomp

                # ir图像通道数转换
                if data_IR.shape[1] == 1:
                    data_IR = data_IR.repeat(1, 3, 1, 1)
                if data_VIS.shape[1] == 1:
                    data_VIS = data_VIS.repeat(1, 3, 1, 1)
                if new_data_Fuse.shape[1] == 1:
                    new_data_Fuse = new_data_Fuse.repeat(1, 3, 1, 1)


                # 数据resize以适应VIT
                data_IR = resize_to_384(data_IR)
                data_VIS = resize_to_384(data_VIS)
                new_data_Fuse = resize_to_384(new_data_Fuse)

                # 奖励模型进行打分
                new_result = reward_Model(new_data_Fuse, data_IR, data_VIS)
                scores = new_result["scores"]
                heatmaps = new_result.pop('heatmaps')
                white_ratio = compute_white_ratio_batch(heatmaps, threshold=0.5)

                # reward_loss = - (1+white_ratio)*scores["Artifacts"] - scores["Texture_Preservation"] - scores["Thermal_Retention"] - scores["Sharpness"]
                reward_loss = (
                    - w_artifact * (1 + white_ratio) * scores["Artifacts"] 
                    - w_texture * scores["Texture_Preservation"] 
                    - w_thermal * scores["Thermal_Retention"] 
                    - w_sharp * scores["Sharpness"]
                )
                reward_loss = reward_loss.mean()
                # reward_loss = 0.1*reward_loss
                reward_loss = reward_loss_weight * reward_loss


                pseudo_fused_images = pseudo_fuse_cuda(data_IR, data_VIS)
                pseudo_fusion_loss = MSELoss(new_data_Fuse, pseudo_fused_images)
                # pseudo_fusion_loss = 100*pseudo_fusion_loss 
                pseudo_fusion_loss = pseudo_fusion_weight * pseudo_fusion_loss
                loss = reward_loss + pseudo_fusion_loss + loss1
                #loss = reward_loss + 2.0*pseudo_fusion_loss + loss1
                # loss = reward_loss + 1.5*pseudo_fusion_loss + loss1

                # loss = reward_loss + 3*loss1
                # print(f"总损失: {loss.item():.4f}")

                # 记录损失和分数
                # if i :
                if i % 10 == 0:
                    logging.info(f" 全图损失 full_image_loss = {reward_loss.mean().item():.4f}, " +
                                 f" 融合损失 loss1 = {loss1.mean().item():.4f}, " +
                                f" 融合图像损失 pseudo_fusion_loss = {pseudo_fusion_loss.mean().item():.4f}, " )
            
            loss.backward()
            optimizer.step()
            # 3. 用 scaler 进行反向、step、update
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            # 4. scheduler（可以放在每 batch 或每 epoch）
            scheduler.step()

            # Determine approximate time left
            batches_done = epoch * len(trainloader) + i
            batches_left = num_epochs * len(trainloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            logging.info(
                f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(trainloader)}] [loss: {loss.item():.6f}] ETA: {str(time_left)[:10]}"
            )

        # if (epoch + 1) % 5 == 0:  
        # if (epoch + 1) == 3 or (epoch + 1) == 5:
        if (epoch):
            checkpoint = {
                'DE_Encoder': Encoder.state_dict(),
                'DE_Decoder': Decoder.state_dict(),
                'LowFreqExtractor': LowFreqExtractor.state_dict(),
                'HighFreqExtractor': HighFreqExtractor.state_dict()
            }
            # 保存新文件
            save_path = f"DCEvo_epoch{epoch+1}_{timestamp}.pth"
            save_path = os.path.join("models/DCEvo", save_path)
            torch.save(checkpoint, save_path)
            logging.info(f"Model saved at epoch {epoch+1} to {save_path}")

            scheduler.step()
if True:
    checkpoint = {
        'DE_Encoder': Encoder.state_dict(),
        'DE_Decoder': Decoder.state_dict(),
        'LowFreqExtractor': LowFreqExtractor.state_dict(),
        'HighFreqExtractor': HighFreqExtractor.state_dict()
    }
    torch.save(checkpoint, os.path.join("models/DCEvo/DCEvo"+timestamp+'.pth'))