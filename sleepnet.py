
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import numbers


class CrossAttention(nn.Module):
    def __init__(self, fudim, detdim, num_heads, bias, sf=2):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.k = nn.Sequential(nn.Upsample(scale_factor=sf, mode='nearest'), # for detfm Upsampling
                               nn.Conv2d(detdim, fudim, kernel_size=1, stride=1, bias=bias),
                               nn.SiLU(inplace=True),
                               nn.Conv2d(fudim, fudim, kernel_size=int(sf*2-1), stride=1, padding=sf-1, groups=fudim, bias=bias))
        self.q = nn.Sequential(nn.Conv2d(fudim, fudim, kernel_size=1, stride=1, bias=bias), # for fufm Downsampling
                                nn.SiLU(inplace=True),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(fudim, fudim, kernel_size=3, stride=2, padding=0, groups=fudim, bias=bias),
                                nn.SiLU(inplace=True),
                                nn.Conv2d(fudim, fudim, kernel_size=1, stride=1, bias=bias),
                                nn.SiLU(inplace=True),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(fudim, fudim, kernel_size=3, stride=2, padding=0, groups=fudim, bias=bias))
        self.v = nn.Sequential(nn.Conv2d(fudim, fudim, kernel_size=1, stride=1, bias=bias), # for fufm value
                                nn.SiLU(inplace=True),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(fudim, fudim, kernel_size=3, stride=1, padding=0, groups=fudim, bias=bias))
        self.project_out = nn.Conv2d(fudim, fudim, kernel_size=1, bias=bias)

    def forward(self, fux, detx):
        b, c, h, w = fux.shape
        dsh = int(h * 0.25)
        dsw = int(w * 0.25)
        k = self.k(detx)  # [b,fudim,fuh,fuw]
        q = self.q(fux)
        v = self.v(fux)
        q = rearrange(q, 'b (head c) dsh dsw -> b head c (dsh dsw)',
                    head=self.num_heads)
        k = rearrange(k, 'b (head c) dsh dsw -> b head c (dsh dsw)',
                    head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                    head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(self,
                dimf=64,
                dimd=128,
                num_heads=8,
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type='WithBias',
                sf=2):
        super(CrossAttentionBlock, self).__init__()

        self.norm1f = LayerNorm(dimf, LayerNorm_type)
        self.norm1d = LayerNorm(dimd, LayerNorm_type)
        self.attn = CrossAttention(dimf, dimd, num_heads, bias, sf)
        self.norm2 = LayerNorm(dimf, LayerNorm_type)
        self.ffn = FeedForward(dimf, ffn_expansion_factor, bias)
        self.pmt = prominent()
        self.selfatt = TransformerBlock(dim=dimf, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, fux, detx):
        x = (self.attn(self.norm1f(fux), self.norm1d(detx))) + fux
        x = x + self.pmt(self.ffn(self.norm2(x)))

        return self.selfatt(x)
        return x


class DFFeatureEncoder(nn.Module):
    def __init__(self,
                inp_channels=1,
                out_channels=1,
                dimf=64,
                dimd=128,
                num_blocks=5,
                heads=[8, 8, 8],
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type='WithBias',
                sf=2
                ):

        super(DFFeatureEncoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dimf)

        self.encoder_level1 = nn.Sequential(*[LowFreqExtractor(dim=dimf) for i in range(num_blocks)])

        self.baseFeature = LowFreqExtractor(dim=dimf)

        self.detailFeature = HighFreqExtractor()

    def forward(self, inp_img):    # , in_det_img
        inp_enc_level1 = self.patch_embed(inp_img)            # [b, dim, 96, 96]
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # [b, dim, 96, 96]
        base_feature = self.baseFeature(out_enc_level1)       # [b, dim, 96, 96]
        detail_feature = self.detailFeature(out_enc_level1)   # [b, dim, 96, 96]
        return (base_feature,
                detail_feature,
                out_enc_level1)
        return (base_feature+(self.crossatt[0](base_feature, in_det_img)),
                (detail_feature+self.crossatt[1](detail_feature, in_det_img)),
                (out_enc_level1+self.crossatt[2](out_enc_level1, in_det_img)))



class AttentionBase(nn.Module):
    def __init__(self,
                dim,
                num_heads=8,
                qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.proj(out)
        return out

class Mlp(nn.Module):
    def __init__(self,
                in_features,
                hidden_features=None,
                ffn_expansion_factor = 2,
                bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)
        self.project_in = nn.Conv2d(in_features, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class prominent(nn.Module):
    def __init__(self, channels=0):
        super(prominent, self).__init__()
        self.act = nn.Sigmoid()
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.act(x)
        return self.act2(x * self.act(y.pow(2) / y.mean(dim=[2, 3], keepdim=True)))


class LowFreqExtractor(nn.Module):
    def __init__(self, dim,):
        super(LowFreqExtractor, self).__init__()
        self.net1 = C2f(c1=dim, c2=dim, n=6, g=int(dim//2), e=0.5)
        self.net2 = C2f(c1=dim, c2=dim, n=6, g=int(dim//2), e=0.5)

    def forward(self, x):
        return self.net2(self.net1(x))


class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


class DWBlock(nn.Module):
    def __init__(self, inp, oup,):
        super(DWBlock, self).__init__()
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(inp, inp, 1, groups=1, bias=False),
            nn.ReLU6(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(inp, inp, 3, groups=inp, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inp, inp, 1, groups=1, bias=False),
            nn.ReLU6(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(inp, oup, 3, groups=inp, bias=False),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)    # SHOULD BE NO RESIDUAL CONNECTION

class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        self.theta_phi = DWBlock(inp=32, oup=32)
        self.theta_rho = DWBlock(inp=32, oup=32)
        self.theta_eta = DWBlock(inp=32, oup=32)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)
        self.pmt = simam_module()

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return self.pmt(z1), self.pmt(z2)

class HighFreqExtractor(nn.Module):
    def __init__(self, num_layers=3):
        super(HighFreqExtractor, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU(inplace=True)  # default activation
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act((self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=False, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.att = simam_module(c2)

    def forward(self, x):
        return x + (self.cv2(self.cv1(x))) if self.add else (self.cv2(self.cv1(x)))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=2, shortcut=True, g=1, e=0.5, partnum=1):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.att = prominent(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return x + self.att(self.cv2(torch.cat(y, 1)))


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5)                  # * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5)+ self.bias           # * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                    head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                    head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                    head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, pn=1):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.att1 = prominent(dim)
        self.att2 = prominent(dim)

    def forward(self, x):
        x = x + self.att1(self.attn(self.norm1(x)))
        x = x + self.att2(self.ffn(self.norm2(x)))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                            stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class DE_Encoder(nn.Module):
    def __init__(self,
                inp_channels=1,
                out_channels=1,
                dim=64,
                num_blocks=5,
                heads=[8, 8, 8],
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type='WithBias',
                ):
        super(DE_Encoder, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[LowFreqExtractor(dim=dim) for i in range(num_blocks)])
        self.baseFeature = LowFreqExtractor(dim=dim)
        self.detailFeature = HighFreqExtractor()

    def forward(self, inp_img):
        embedded_feature = self.patch_embed(inp_img)
        base_feature = self.encoder_level1(embedded_feature)
        lf_feature = self.baseFeature(base_feature)
        hf_feature = self.detailFeature(base_feature)
        return lf_feature, hf_feature, base_feature


class DE_Decoder(nn.Module):
    def __init__(self,
                inp_channels=1,
                out_channels=1,
                dim=64,
                num_blocks=5,
                heads=[8, 8, 8],
                ffn_expansion_factor=2,
                bias=False,
                LayerNorm_type='WithBias',
                ):

        super(DE_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[LowFreqExtractor(dim=dim) for i in range(num_blocks)])
        self.output = nn.Sequential(
            TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
            bias=bias, LayerNorm_type=LayerNorm_type),
            nn.ReflectionPad2d(1),
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                    stride=1, padding=0, bias=bias),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                    stride=1, padding=0, bias=bias),
            )
        self.sigmoid = nn.Sigmoid()
    def forward(self, inp_img, base_feature, detail_feature, return_logits=False):
        embedded_feature = torch.cat((base_feature, detail_feature), dim=1)
        embedded_feature = self.reduce_channel(embedded_feature)
        base_feature = self.encoder_level2(embedded_feature)
        if inp_img is not None:
            # A tensor's leading axis is the batch; only a list/tuple holds modalities.
            residual = torch.stack(inp_img).mean(0) if isinstance(inp_img, (list, tuple)) else inp_img
            out = self.output(base_feature) + residual
        else:
            out = self.output(base_feature)
        return (out if return_logits else self.sigmoid(out)), base_feature
