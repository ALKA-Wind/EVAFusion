import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim

from sleepnet import (DE_Encoder, DE_Decoder, LowFreqExtractor, HighFreqExtractor,)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
from torchvision import transforms as T


def calculate_cc(img1, img2):
    """Calculate Correlation Coefficient between two images"""
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    mean1 = np.mean(img1_flat)
    mean2 = np.mean(img2_flat)
    
    numerator = np.sum((img1_flat - mean1) * (img2_flat - mean2))
    denominator = np.sqrt(np.sum((img1_flat - mean1)**2) * np.sum((img2_flat - mean2)**2))
    
    if denominator == 0:
        return 0
    return numerator / denominator


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr





def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    return ssim(img1, img2, data_range=255)


def fusion_metrics(img_fused, img_ir, img_vis):
    """
    Calculate fusion metrics (CC, PSNR, SSIM)
    Returns: dict with all metric values
    """
    # Ensure images are in the right format (0-255, uint8)
    img_fused = np.clip(img_fused, 0, 255).astype(np.uint8)
    img_ir = np.clip(img_ir, 0, 255).astype(np.uint8)
    img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)
    
    metrics = {}
    
    # CC with both source images
    cc_ir = calculate_cc(img_fused, img_ir)
    cc_vis = calculate_cc(img_fused, img_vis)
    metrics['CC'] = (cc_ir + cc_vis) / 2  # Average CC
    
    # PSNR with both source images
    psnr_ir = calculate_psnr(img_fused, img_ir)
    psnr_vis = calculate_psnr(img_fused, img_vis)
    metrics['PSNR'] = (psnr_ir + psnr_vis) / 2  # Average PSNR
    
    # SSIM with both source images
    ssim_ir = calculate_ssim(img_fused, img_ir)
    ssim_vis = calculate_ssim(img_fused, img_vis)
    metrics['SSIM'] = (ssim_ir + ssim_vis) / 2  # Average SSIM
    
    return metrics


def test(pth_path='', out_path=''):
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.CRITICAL)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # ckpt_path=pth_path
    ckpt_path= "models/DCEvo/DCEvo_epoch10_2025-07-25_07-47-21.pth"
    
    # Dictionary to store all metrics for all datasets
    all_results = {}
    
    # for dataset_name in ["M3FD", "FMB", "TNO", "RoadScene"]:
    for dataset_name in [ "M3FD","TNO", "RoadScene"]:
        print("="*50)
        print(f"Testing on dataset: {dataset_name}")
        print("="*50)
        
        test_folder=os.path.join('datasets/',dataset_name) 
        # test_out_folder=os.path.join('datasets/', dataset_name, 'DCEvo_Fusion_Only')
        test_out_folder=os.path.join('datasets/', dataset_name, 'fusion0725')
        # test_out_folder=os.path.join('result/05101032/', dataset_name, 'images')
    
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Encoder = nn.DataParallel(DE_Encoder()).to(device)
        Decoder = nn.DataParallel(DE_Decoder()).to(device)
        LFExtractor = nn.DataParallel(LowFreqExtractor(dim=64)).to(device)
        HFExtractor = nn.DataParallel(HighFreqExtractor(num_layers=3)).to(device)
    
        Encoder.load_state_dict(torch.load(ckpt_path)['DE_Encoder'])
        Decoder.load_state_dict(torch.load(ckpt_path)['DE_Decoder'])
        LFExtractor.load_state_dict(torch.load(ckpt_path)['LowFreqExtractor'])
        HFExtractor.load_state_dict(torch.load(ckpt_path)['HighFreqExtractor'])
        
        Encoder.eval()
        Decoder.eval()
        LFExtractor.eval()
        HFExtractor.eval()   
        transform = T.Compose([T.Resize((768, 1024)), T.ToTensor()])
        
        # Initialize metrics storage for this dataset
        dataset_metrics = {
            'CC': [],
            'PSNR': [],
            'SSIM': []
        }
        
        img_list = os.listdir(os.path.join(test_folder,"ir"))
        print(f"Processing {len(img_list)} images...")
    
        with torch.no_grad():
            for idx, img_name in enumerate(tqdm(img_list, desc="Processing images")):
                # Load original images for metric calculation
                original_ir = image_read_cv2(os.path.join(test_folder,"ir",img_name), mode='GRAY')
                original_vis = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')
    
                data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
                data_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
    
                data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
                data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
                
                feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
                feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
                
                feature_F_B = LFExtractor(feature_I_B+feature_V_B)
                feature_F_D = HFExtractor(feature_I_D+feature_V_D)
                
                data_Fuse, feature_F = Decoder(data_VIS*0.5+data_IR*0.5, feature_F_B, feature_F_D) 
                
                fi = np.squeeze((data_Fuse * 255).cpu().numpy())
                
                # Calculate metrics
                metrics = fusion_metrics(fi, original_ir, original_vis)
                
                # Store metrics
                for metric_name, value in metrics.items():
                    dataset_metrics[metric_name].append(value)
                
                # Save fused image
                img_save(fi, img_name.split(sep='.')[0], test_out_folder)
        
        # Calculate average metrics for this dataset
        avg_metrics = {}
        for metric_name, values in dataset_metrics.items():
            avg_metrics[metric_name] = np.mean(values)
            avg_metrics[f'{metric_name}_std'] = np.std(values)
        
        all_results[dataset_name] = avg_metrics
        
        print(f"\nResults for {dataset_name}:")
        print("-" * 30)
        print(f"CC:   {avg_metrics['CC']:.4f} ± {avg_metrics['CC_std']:.4f}")
        print(f"PSNR: {avg_metrics['PSNR']:.4f} ± {avg_metrics['PSNR_std']:.4f}")
        print(f"SSIM: {avg_metrics['SSIM']:.4f} ± {avg_metrics['SSIM_std']:.4f}")
        print(f"\n{len(img_list)} results have been saved in {test_out_folder}")
    
    # Print summary table
    print("\n" + "="*65)
    print("SUMMARY OF ALL DATASETS")
    print("="*65)
    print(f"{'Dataset':<12} {'CC':<12} {'PSNR':<12} {'SSIM':<12}")
    print("-" * 65)
    
    for dataset_name, metrics in all_results.items():
        print(f"{dataset_name:<12} {metrics['CC']:<12.4f} {metrics['PSNR']:<12.4f} {metrics['SSIM']:<12.4f}")
    
    # Save results to JSON file
    results_file = "zhibiao/0725.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nDetailed results saved to: {results_file}")
        

if __name__ == '__main__':
    test()