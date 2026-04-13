# EVAFusion

Jinyuan Liu, Xingyuan Li, Qingyun Mei, Haoyuan Xu, Zhiying Jiang, Long Ma, Risheng Liu, Xin Fan, **"Bridging Human Evaluation to Infrared and Visible Image Fusion"**,
IEEE/CVF Conference on Computer Vision and Pattern Recognition **(CVPR)**, 2026.

![Abstract](Figures/pipeline.jpg)

## Environment
```
conda env create -f environment.yml
```

## Dataset Show
This figure overviews the collected dataset, highlighting its data, label, and scene diversity.

The EVAFusion dataset is publicly available at Hugging Face Datasets:
👉 [https://huggingface.co/datasets/HHOODD/EVAFusion_dataset](sslocal://flow/file_open?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FHHOODD%2FEVAFusion_dataset&flow_extra=eyJsaW5rX3R5cGUiOiJjb2RlX2ludGVycHJldGVyIn0=)

![Abstract](Figures/data_show.jpg)

#  Model Training

EVAFusion training consists of **two stages**:

---

# Stage 1: Train Reward Model

The Reward Model is fine‑tuned based on:

```
google/vit-large-patch16-384
```

## Step 1: Download Pretrained ViT Model

Please download:

```
google/vit-large-patch16-384
```

and place it into the following directory:

```
RichHF/
```

Directory example:

```
RichHF/
└── vit-large-patch16-384
```


## Step 2: Train Reward Model

After preparing the pretrained model and dataset, run:

```bash
python train_reward.py
```

This step will fine‑tune the Reward Model using the **EVAFusion dataset**.

---

# Stage 2: Fine‑Tune IVIF Model with Reward Model

After the Reward Model is trained, run the following script to fine‑tune the final IVIF fusion model:

```bash
python reward_final.py
```

This stage uses the trained Reward Model to guide the fusion network optimization.

The trained IVIF model will be saved automatically to:

```
checkpoints/
```

#  Inference

After training, run the following command for fusion inference:

```bash
python test_Fusion.py
```

## Fusion Results
1. Quantitative comparison of infrared and visible image fusion between our DCEvo and state-of-the-art methods on M3FD, RoadScene, TNO and FMB datasets.


![Abstract](Figures/Quantitative_comparison.jpg)

3. Qualitative comparisons of our DCEvo and existing image fusion methods. From top to bottom: low-light in TNO, high-brightness in RoadScene and low-quality in M3FD..

![Abstract](Figures/Qualitative_comparison.jpg)

## Citation
```
@article{liu2026bridging,
  title={Bridging Human Evaluation to Infrared and Visible Image Fusion},
  author={Liu, Jinyuan and Li, Xingyuan and Mei, Qingyun and Xu, Haoyuan and Jiang, Zhiying and Ma, Long and Liu, Risheng and Fan, Xin},
  journal={arXiv preprint arXiv:2603.03871},
  year={2026}
}
```
