"""Shared DCEvo forward pass and checkpoint IO, preserving the original state keys."""
from pathlib import Path

import torch
from torch import nn
from sleepnet import DE_Encoder, DE_Decoder, LowFreqExtractor, HighFreqExtractor


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = DE_Encoder()
        self.Decoder = DE_Decoder()
        self.LowFreqExtractor = LowFreqExtractor(dim=64)
        self.HighFreqExtractor = HighFreqExtractor(num_layers=3)

    def forward(self, ir, vi, return_features=False, return_logits=False):
        vb, vd, _ = self.Encoder(vi)
        ib, id_, _ = self.Encoder(ir)
        fused, _ = self.Decoder((vi + ir) * .5,
                                self.LowFreqExtractor(vb + ib),
                                self.HighFreqExtractor(vd + id_), return_logits=return_logits)
        return (fused, (vb, ib, vd, id_)) if return_features else fused

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = cls()
        for key, module in model.components().items():
            if key not in checkpoint:
                raise ValueError(f'Checkpoint is missing {key}: {path}')
            state = {k.removeprefix('module.'): v for k, v in checkpoint[key].items()}
            module.load_state_dict(state, strict=True)
        return model.to(device)

    def components(self):
        return {'DE_Encoder': self.Encoder, 'DE_Decoder': self.Decoder,
                'LowFreqExtractor': self.LowFreqExtractor, 'HighFreqExtractor': self.HighFreqExtractor}

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({key: {k: v.detach().cpu() for k, v in module.state_dict().items()}
                    for key, module in self.components().items()}, path)
