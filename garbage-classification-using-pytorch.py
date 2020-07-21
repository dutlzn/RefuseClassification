# 1 导入库
## 系统库
import os 
from os import walk
## torch
import torch
import torch.nn as nn
from torchvision import datasets 
## 相关参数
from args import args
## 数据的预处理
from transform import preprocess
## 模型pre_trained_model 加载、训练、评估、标签映射关系
from model import train, evaluate, initital_model, class_idname
## 工具类： 日志类工具类、模型保存、优化器 （别人提供的 需要重写）
from utils.logger import Logger 
from utils.misc import save_checkpoint, get_optimizer
## 训练矩阵效果评估工具类
from sklearn import metrics
# 2 数据探测
base_path = './data/garbage-classify-for-pytorch'
# for (dirpath, dirnames, filenames) in os.walk(base_path):
#     if len(filenames)>0:
#         print("*"*60)
#         print('Director path:', dirpath)
#         print('Total examples:', len(filenames))
#         print('File name Examples:', filenames[:5])

# 3 数据封装 ImageFolder 格式
TRAIN = "{}/train".format(base_path)
VALID = "{}/val".format(base_path)
# print("train_data_path:", TRAIN)
# print("val data path:", VALID)

# root (string): Root directory path.
# transform (callable, optional): A function/transform that  takes in an PIL image
# and returns a transformed version. E.g, ``transforms.RandomCrop``
train_data = datasets.ImageFolder(
    root = TRAIN,
    transform = preprocess
)
val_data = datasets.ImageFolder(
    root = VALID,
    transform = preprocess
)

assert train_data.class_to_idx.keys() == val_data.class_to_idx.keys()

# print('img:', train_data.imgs[:2])

# 4 批量数据加载
batch_size = 2
num_workers = 2
train_loader = torch.utils.data.DataLoader(
    train_data, 
    batch_size=batch_size, 
    num_workers = num_workers,
    shuffle=True)
val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size, 
    num_workers = num_workers,
    shuffle=False)

# 5 定义模型训练和验证方法



# 入口程序
if __name__ == '__main__':
    # image, label = next(iter(train_loader))
    # print(label)
    # print(image.shape)
    # print("test")

    # 模型初始化
    model_name = args.model_name
    num_class = args.num_classes
    initital_model(model_name, num_class, feature_extract=True)