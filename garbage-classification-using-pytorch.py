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
from model import train, evaluate, initital_model, class_id2name
## 工具类： 日志类工具类、模型保存、优化器 （别人提供的 需要重写）
from utils.logger import Logger 
from utils.misc import save_checkpoint, get_optimizer
## 训练矩阵效果评估工具类
from sklearn import metrics

# 1 获取所有的参数
state = {k:v for k,v in args._get_kwargs()}
print("all args:", state)
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

# 定义全局变量，保存准确率
best_acc = 0 
# 5 定义模型训练和验证方法
def run(model, train_loader, val_loader):
    '''
    模型训练和预测
    :param model: 初始化的model
    :param train_loader: 训练数据
    :param val_loader: 验证数据
    :return:
    '''
    ## 初始化变量
    ### 模型保存的变量
    global best_acc
    ### 训练c个类别的分类问题，我们CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    ### torch.optim 是一个各种优化算法库 
    ### optimizer对象能保存当前的参数状态并且基于计算梯度更新参数
    optimizer = get_optimizer(model, args)

    ## 加载checkpoint: 断点续传，可以指定迭代的开始位置进行重新训练
    ## 评估：混淆矩阵 准确率 召回率 F1-score
    ## 模型的训练和验证
    for epoch in range(state['start_epoch'],state['epochs']+1):
        print('[{}/{}] Training'.format(epoch,state['epochs']))
        # train
        train_loss,train_acc = train(train_loader, model, criterion, optimizer)
        # val
        test_loss,test_acc = evaluate(val_loader,model,criterion,test=None)

        # 核心参数保存logger

        # 保存模型
        is_best = test_acc > best_acc
        best_acc = max(test_acc,best_acc)
        save_checkpoint({
            'epoch':epoch + 1,
            'state_dict':model.state_dict(),
            'train_acc':train_acc,
            'test_acc':test_acc,
            'best_acc':best_acc,
            'optimizer':optimizer.state_dict()

        }, is_best, checkpoint=args.checkpoint)

        print('Best acc:', best_acc)
        # print(best_acc)

    pass 

# 入口程序
if __name__ == '__main__':
    pass 
    # image, label = next(iter(train_loader))
    # print(label)
    # print(image.shape)
    # print("test")

    # 模型初始化
    model_name = args.model_name
    num_class = args.num_classes
    model_ft =  initital_model(model_name, num_class, feature_extract=True)

    # 模型训练和评估
    run(model_ft, train_loader, val_loader)