'''
Date: 2021-11-25 21:49:19
LastEditors: HowsenFisher
LastEditTime: 2021-11-27 23:20:23
FilePath: \GAN\train.py
'''
import torch
from Utils.Logger import logger
from tqdm import tqdm
from Model.GAN import GAN
from torchvision.datasets import mnist  # 导入pytorch内置数据
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
# IO
from glob import glob
import os
# DDP相关
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
# 命令行参数
import argparse
import yaml


######################################################################################################
"""
	创建解析器
"""
parser = argparse.ArgumentParser(
    description='Train a GAN model.'
)
# 添加local_rank命令行参数（DDP模式下必须有这个参数）
parser.add_argument("--useDDP", default=0, type=int)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--config_path", default="./Config/default.yml", type=str)
# 拿出参数集
args = parser.parse_args()
useDDP = args.useDDP
#######################################################################################################
"""
    DDP初始化：要放在所有DDP代码前
"""
if useDDP:
    # nccl是GPU设备上最快、最推荐的后端
    dist.init_process_group(backend='nccl')
    # local_rank参数
    local_rank = dist.get_rank()
    # DDP：DDP backend初始化
    torch.cuda.set_device(local_rank)
else:
    local_rank = 0
#######################################################################################################
"""
	日志
"""
# 日志生成器
if useDDP:
    logger = logger(dist.get_rank())
else:
    logger = logger(0)
#######################################################################################################
"""
	GPU检查及环境
"""
logger.info("检查环境")
# 判断GPU是否可用
use_gpu = torch.cuda.is_available()
logger.info("GPU是否可用: %s" % use_gpu)
# 如果可用，输出GPU个数
if use_gpu:
    gpu_c = torch.cuda.device_count()
    logger.info("GPU数量:%d" % gpu_c)
# 如果cuda可以使用，device为cuda，否则是cpu
device = torch.device("cuda" if use_gpu else "cpu")
# 没有out文件夹就创建
if not os.path.exists("./out"):
    try:
        os.mkdir("./out")
    except OSError:
        pass
#######################################################################################################
"""
	参数
"""
logger.info("读取设置文件")
cfg = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
# 超参数
img_size = cfg["model"]["img_size"]  # 图片尺寸
latent_size = cfg["model"]["latent_size"]     # 潜在空间code的尺寸
# 训练参数
batch_size = cfg["train"]["batch_size"]     # 批大小
epochs = cfg["train"]["epochs"]       # 训练轮数
logger.info("设置文件加载完成")
#######################################################################################################
"""
	生成模型
"""
# 如果gpu可用，将模型放到GPU上
logger.info("正在生成GAN模型(with %s)" % ("gpu" if use_gpu else "cpu"))
gan = GAN(latent_size, img_size).to(device)
# 如果是主进程，模型转换为DDP模式
if local_rank is not None and not local_rank == -1 and useDDP:
    gan = DDP(gan, device_ids=[local_rank], output_device=local_rank)
    gan = gan.module
logger.info("打印模型")
# 只有主进程打印模型
if local_rank == 0:
    gan.PrintNet(logger)
######################################################################################################
"""
	数据集
"""
# 载入数据集
logger.info("正在载入Mnist数据集")
# 如果不存在MNIST数据集则下载，否则进行读取
train_set = mnist.MNIST('./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
]))
logger.info("成功载入Mnist数据集")

if useDDP:
    # DDP：使用DistributedSampler
    train_sampler = DistributedSampler(train_set)
    logger.info("制作dataLoader")
    # 制作DDP版本的dataloader
    train_data_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, sampler=train_sampler)
else:
    train_data_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True)
logger.info("dataLoader载入完毕")
#########################################################################################################
"""
	一些函数
"""
def getTaskId():
    taskId = len(glob("./out/*"))+1
    try:
        os.mkdir("./out/task%d" % taskId)
    except OSError:
        pass
    return taskId

logger.info("创建新任务")
# 只有主进程会新建task
if local_rank == 0:
    taskId = getTaskId()
    logger.info("新任务号: %d"%taskId)

def showGernerateRes(ep, m, n):
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
        axs[index//n, index % n].imshow(img, cmap='gray')
        axs[index//n, index % n].axis('off')
    # 保存图片
    plt.savefig("./out/task%d/%d.png" % (taskId, ep))
    # 释放内存
    plt.close()


############################################################################################################
"""
	训练
"""
logger.info("开始训练")
# 创造real label（全是1）和fake label（全是0）
real_labels = torch.ones(batch_size, 1).to(device)
fake_labels = torch.zeros(batch_size, 1).to(device)
# 每一轮
for epoch in range(epochs):
    # 打印轮次
    logger.info("epoch {}/{}".format(epoch+1, epochs))
    # 主进程控制tqdm进度条
    if local_rank == 0:
        train_data_loader = tqdm(train_data_loader)
    # 非主进程sampler传入轮次信息
    else:
        train_data_loader.sampler.set_epoch(epoch)
    # 每一批
    for index, (real_img, target) in enumerate(train_data_loader):
        # 为了防止最后一批图片数量不足batchsize，要去修改real_labels
        if target.shape[0] != batch_size:
            real_labels = torch.ones(target.shape[0], 1).to(device)
        ############################################################################################################
        # 训练判别器
        real_img = real_img.to(device)
        # 让判别器先学习一个真图片
        # 判别器输出判定
        real_output = gan.discriminator(real_img)
        # 计算判定的损失
        Dloss1 = gan.discriminator.LossFunc()(real_output, real_labels)

        # 再让判别器学习一个假图片
        # 随机生成一个[batchsize,1000]的code
        code = torch.randn(batch_size, latent_size).to(device)
        # 让生成器生成一张图片，标签是假
        generate_img = gan.generator(code).to(device)
        # 判别器输出判定
        output = gan.discriminator(generate_img)
        # 判别器根据标签和自己的判定，确定损失
        Dloss2 = gan.discriminator.LossFunc()(output, fake_labels)
        # loss加和
        Dloss = Dloss1 + Dloss2
        # 清除梯度
        gan.discriminator.optimizer.zero_grad()
        gan.generator.optimizer.zero_grad()
        # 做Loss反向传播
        Dloss.backward()
        # 更新判别器参数
        gan.discriminator.optimizer.step()

        ####################################################################################################
        # 训练生成器
        # 随机生成一个[batchsize,1000]的code
        code = torch.randn(batch_size, latent_size).to(device)
        # 生成一张假图片
        generate_img = gan.generator(code)
        # 判别器输出判定结果
        output = gan.discriminator(generate_img)
        # real_label恢复batchsize大小
        real_labels = torch.ones(batch_size, 1).to(device)
        # 生成器要骗过判定器，让它输出real，但实际上有fake，计算loss
        Gloss = gan.generator.LossFunc()(output, real_labels)
        # 清除梯度
        gan.discriminator.optimizer.zero_grad()
        gan.generator.optimizer.zero_grad()
        # 做反向传播
        Gloss.backward()
        # 更新生成器参数
        gan.generator.optimizer.step()
        # 设置进度条右边显示的信息
        if index % 10 == 0 and local_rank == 0:
            train_data_loader.set_postfix(
                Dloss=Dloss.item(), GLoss=Gloss.float().item())

    # 每10轮，主进程保存一次结果图片
    if epoch % cfg["train"]["save_freq"] == 0 and local_rank == 0:
        logger.info("保存模型和结果")
        showGernerateRes(epoch, cfg["out"]["save_size"][0], cfg["out"]["save_size"][1])
        torch.save(gan.state_dict(), "./out/task%d/%d.pt" % (taskId, epoch))
