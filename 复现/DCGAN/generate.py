'''
Date: 2021-11-28 20:18:48
LastEditors: HowsenFisher
LastEditTime: 2021-11-28 20:39:31
FilePath: \DCGAN\generate.py
'''
import torch
from Model.GAN import GAN
import yaml
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms

parser = argparse.ArgumentParser(
    description='Train a GAN model.'
)

parser.add_argument("--config_path", default="./Config/default.yml", type=str)
# 拿出参数集
args = parser.parse_args()
# 判断GPU是否可用
use_gpu = torch.cuda.is_available()
# 如果cuda可以使用，device为cuda，否则是cpu
device = torch.device("cuda" if use_gpu else "cpu")

cfg = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
# 超参数
img_size = cfg["model"]["img_size"]  # 图片尺寸
latent_size = cfg["model"]["latent_size"]     # 潜在空间code的尺寸

gan = GAN(latent_size, img_size).to(device)
gan.load_state_dict(torch.load("420.pt"))

def showGernerateRes(m, n):
    """
            保存1次生成的图
    """
    # 随机生成m*n个latent code
    code = torch.randn(m*n, latent_size).to(device)
    # 送入生成器输出m*n张图
    tensor = gan.generator(code)
    # 建立plt子图m*n个，每个大小4*4
    fig, axs = plt.subplots(m, n, figsize=(4, 4), sharey=True, sharex=True)
    # 遍历每一个图片
    for index, img in enumerate(tensor):
        # 转换为PIL图片
        img = transforms.ToPILImage()(img)
        # 子图绘制灰度图
        axs[index//n, index % n].imshow(img)
        axs[index//n, index % n].axis('off')
    # 保存图片
    plt.savefig("./res.png")
    # 释放内存
    plt.close()

showGernerateRes(4, 4)