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

# 3 数据封装

# 4 批量数据加载

# 5 定义模型训练和验证方法



# 入口程序
if __name__ == '__main__':
    print("hello")