import torch
from model import preprocess_image, RAHF
from PIL import Image

fused_path = "test/fused/00083.png"
ir_path = "test/ir/00083.png"
vi_path = "test/vi/00083.png"

fused_tensor = preprocess_image(fused_path)
ir_tensor = preprocess_image(ir_path)
vi_tensor = preprocess_image(vi_path)

model = RAHF()
# 注意路径问题，win和Linux不一样
ckpt_path = 'exp/20250418_154258/model.pt'

# 指定显卡序号
model.load_state_dict(torch.load(ckpt_path, map_location='cuda:1'))
model.eval()

# 禁用梯度计算
with torch.no_grad():
    out = model(fused_tensor, ir_tensor, vi_tensor)
heatmaps = out.pop('heatmaps')
print(out)

# save the heatmaps
for k, map_ in heatmaps.items():
    imgs = map_.detach().cpu()
    imgs = imgs.permute([1, 2, 0])  # change to h x w x 1
    imgs = imgs.expand([-1, -1, 3])
    imgs = imgs * 255
    im = Image.fromarray(imgs.numpy().astype('uint8'))
    im.save(f'{k}.jpg')