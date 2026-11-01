import torch
import torch.nn.functional as F

def vsm_cuda(img):
    """
    在 CUDA (GPU) 上计算图像的视觉显著性图 (VSM).

    Args:
        img (torch.Tensor): 灰度图像，形状为 [B, 1, H, W], dtype=torch.uint8.

    Returns:
        torch.Tensor: 显著性图, 形状为 [B, 1, H, W], dtype=torch.float32.
    """

    B, C, H, W = img.shape
    his = torch.zeros(B, 256, device=img.device, dtype=torch.int32)

    # 统计直方图 (在 GPU 上)
    for i in range(256):
        mask = (img == i)
        his[:, i] = mask.sum(dim=(1, 2, 3))

    sal = torch.zeros(B, 256, device=img.device, dtype=torch.int32).float() # 确保是 float

    # 计算显著性 (在 GPU 上)
    for i in range(256):
        diff = torch.abs(torch.arange(256, device=img.device) - i)
        sal[:, i] = (diff * his).sum(dim=1).float()

    vsm_map = torch.zeros_like(img, dtype=torch.float32)
    for i in range(256):
        mask = (img == i)
        sal_value = sal[:, i].view(B, 1, 1, 1) # 形状 [B, 1, 1, 1]
        vsm_map = torch.where(mask, sal_value.expand_as(vsm_map), vsm_map)

    # 避免除以零
    max_val = vsm_map.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    max_val = torch.where(max_val == 0, torch.ones_like(max_val), max_val)  # 确保不为 0
    return vsm_map / max_val


def pseudo_fuse_cuda(ir_img, vis_img, c=20):
    """
    在 CUDA (GPU) 上，根据红外和可见光图像的显著性权重生成伪融合图像。

    Args:
        ir_img (torch.Tensor): 红外图像, 形状为 [B, C, H, W].
        vis_img (torch.Tensor): 可见光图像, 形状为 [B, C, H, W].
        c (int, optional): 控制权重分布的参数. Defaults to 20.

    Returns:
        torch.Tensor: 伪融合图像, 形状为 [B, C, H, W].
    """

    # 转换为灰度 (如果需要) - 确保灰度图是 uint8
    if ir_img.shape[1] == 3:
        ir_gray = (0.299*ir_img[:,0] + 0.587*ir_img[:,1] + 0.114*ir_img[:,2]).unsqueeze(1)
    else:
        ir_gray = ir_img.byte()

    if vis_img.shape[1] == 3:
        vis_gray = (0.299 * vis_img[:, 0] + 0.587 * vis_img[:, 1] + 0.114 * vis_img[:, 2]).unsqueeze(1).byte()
    else:
        vis_gray = vis_img.byte()

    # 计算显著性图
    ir_saliency = vsm_cuda(ir_gray)  # 增加通道维度
    vis_saliency = vsm_cuda(vis_gray)

    # 计算权重
    w1 = 0.4 + (ir_saliency * 1 - vis_saliency * 0.4)
    w2 = 1 - w1

    # 执行融合
    fused_image = w1 * ir_img + w2 * vis_img

    return fused_image


if __name__ == '__main__':
    # 示例用法
    B, C, H, W = 2, 3, 128, 128  # 示例批量大小和图像尺寸
    ir_data = torch.rand(B, C, H, W).cuda()
    vis_data = torch.rand(B, C, H, W).cuda()

    fused_gpu = pseudo_fuse_cuda(ir_data, vis_data)
    print("Fused image shape (GPU):", fused_gpu.shape)
    print("Fused image device:", fused_gpu.device)