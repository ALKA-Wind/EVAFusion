import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import ViTModel, AutoImageProcessor
from PIL import Image


class LayerNorm(nn.Module):
    """T5-style LayerNorm over the channel dimension (No bias and no subtraction of mean)."""
    def __init__(self, n_channels):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(n_channels, 1, 1))

    def forward(self, x: torch.Tensor):
        # x is a feature map of shape: batch_size x n_channels x h x w
        var = x.square().mean(dim=1, keepdim=True)
        out = x * (var + 1e-8).rsqrt()
        out = out * self.scale
        return out


class HeatmapPredictor(nn.Module):
    def __init__(self, n_channels):
        super(HeatmapPredictor, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, 768, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            LayerNorm(768),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            LayerNorm(384),
        )

        self.deconv_layers = nn.ModuleList()
        self.conv_layers2 = nn.ModuleList()
        in_channels = 384
        for out_channels in [768, 384, 384, 192]:
            # 上采样特征图
            self.deconv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                    LayerNorm(out_channels),
                    nn.ReLU()
                )
            )
            # 卷积残差块
            self.conv_layers2.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
                    LayerNorm(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
                    LayerNorm(out_channels),
                )
            )
            in_channels = out_channels

        self.relu = nn.ReLU()

        self.last_conv1 = nn.Conv2d(in_channels, 192, kernel_size=3, stride=1, padding='same')
        # relu
        self.last_conv2 = nn.Conv2d(192, 1, kernel_size=3, stride=1, padding='same') # final_channel size
        # sigmoid
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv_layers(x)
        for deconv, conv in zip(self.deconv_layers, self.conv_layers2):
            x = deconv(x)
            identity = x
            x = conv(x)
            x = x + identity
            x = self.relu(x)

        x = self.last_conv1(x)
        x = self.relu(x)
        x = self.last_conv2(x)
        x = self.sigmoid(x)  # (batch_size, 1, height, width)

        output = x.squeeze(1)
        return output


class ScorePredictor(nn.Module):
    def __init__(self, n_channels, n_patches=14*14):
        super(ScorePredictor, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels // 2, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 2),
            nn.ReLU(),
            nn.Conv2d(n_channels // 2, n_channels // 4, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 4),
            nn.ReLU(),
            nn.Conv2d(n_channels // 4, n_channels // 8, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 8),
            nn.ReLU(),
            nn.Conv2d(n_channels // 8, n_channels // 16, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 16),
            nn.ReLU(),
            nn.Conv2d(n_channels // 16, n_channels // 64, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 64),
            nn.ReLU(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(n_channels // 64 * n_patches, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        conv_output = self.conv_layers(x)
        conv_output = conv_output.flatten(1)
        output = self.linear_layers(conv_output)
        return output


class RAHF(nn.Module):
    def __init__(
            self,
            score_types=('Thermal_Retention', 'Texture_Preservation', 'Artifacts', 'Sharpness', 'Overall_Score'),
            heatmap_types=('artifact_heatmap',),
            vit_model="vit-large-patch16-384",

            multi_heads=True,
            # multi_heads=False,
            patch_size=16,
            image_size=384,
        ):
        super(RAHF, self).__init__()
        self.multi_heads = multi_heads
        self.score_types = score_types
        self.heatmap_types = heatmap_types
        self.n_patches = (image_size // patch_size) ** 2
        self.feature_size = image_size // patch_size

        # 初始化共享的ViT模型
        self.vit = ViTModel.from_pretrained(vit_model)
        self.hidden_dim = self.vit.config.hidden_size

        # 特征融合相关
        self.projection = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.vit_fusion = ViTModel(self.vit.config)  # 使用相同配置的新实例

        # 初始化预测头
        if multi_heads:
            self.heatmap_predictor = nn.ModuleDict({
                hm: HeatmapPredictor(self.hidden_dim) for hm in heatmap_types
            })
            self.score_predictor = nn.ModuleDict({
                sc: ScorePredictor(self.hidden_dim, self.n_patches) for sc in score_types
            })
        else:
            self.heatmap_predictor = HeatmapPredictor(self.hidden_dim)
            self.score_predictor = ScorePredictor(self.hidden_dim, self.n_patches)

    
    def forward(self, fused, ir, vi):
        def extract_features(x):
            return self.vit(pixel_values=x).last_hidden_state[:, 1:, :]  # 移除CLS token

        ir_feat = extract_features(ir)    # [B, N, D]
        vi_feat = extract_features(vi)   # [B, N, D]
        fused_feat = extract_features(fused)  # [B, N, D]

        # 特征融合阶段
        combined = torch.cat([ir_feat, vi_feat, fused_feat], dim=-1)  # [B, N, 3D]
        projected = self.projection(combined)  # [B, N, D]
        
        # 添加CLS token并通过ViT融合
        cls_tokens = self.vit_fusion.embeddings.cls_token.expand(projected.size(0), 1, -1)
        sequence = torch.cat([cls_tokens, projected], dim=1)
        sequence += self.vit_fusion.embeddings.position_embeddings[:, :sequence.size(1)]
        fused_output = self.vit_fusion.encoder(sequence).last_hidden_state
        
        # 特征图重构
        feature_map = fused_output[:, 1:].permute(0, 2, 1)  # [B, D, N]
        feature_map = feature_map.view(-1, self.hidden_dim, 
                                     self.feature_size, self.feature_size)

        # 预测阶段
        results = {'heatmaps': {}, 'scores': {}}
        if self.multi_heads:
            for hm in self.heatmap_types:
                results['heatmaps'][hm] = self.heatmap_predictor[hm](feature_map)
            for sc in self.score_types:
                results['scores'][sc] = self.score_predictor[sc](feature_map).squeeze()
        else:
            results['heatmaps'] = {hm: self.heatmap_predictor(feature_map) 
                                  for hm in self.heatmap_types}
            results['scores'] = {sc: self.score_predictor(feature_map).squeeze()
                                for sc in self.score_types}
        return results


def preprocess_image(image_path):
    # transform = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")
    transform = AutoImageProcessor.from_pretrained("vit-large-patch16-384")
    # image = Image.open(image_path).convert("RGB")
    image = Image.open(image_path)
    # 将图像进行处理，返回pytorch张量，提取张量数据，添加批次维度
    return transform(image, return_tensors="pt")['pixel_values'][0].unsqueeze(0)


if __name__ == "__main__":
    # Example usage
    image_path = "data/a.jpg"
    caption = "A description of the image"

    image_tensor = preprocess_image(image_path)
    model = RAHF()

    out = model(image_tensor, caption, target_text=None)
