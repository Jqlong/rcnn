import copy
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import alexnet

from utils.custom_batch_sampler import CustomBatchSampler
from utils.custom_classifier_dataset import CustomClassifierDataset
from utils.pascal_voc_car import check_dir


def load_data(data_root_dir):
    # 数据预处理
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((227, 227)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # 准备一个字典来存储数据
    data_loaders = {}
    data_sizes = {}
    remain_negative_list = list()

    # 获取训练数据和验证数据
    for name in ["train", "val"]:
        data_dir = os.path.join(data_root_dir, name)
        # 加载数据
        data_set = CustomClassifierDataset(data_dir, transform=transform)
        data_sampler = CustomBatchSampler(data_set.get_positive_num(), data_set.get_negative_num(), 32, 96)
        data_loader = DataLoader(data_set, batch_size=128, sampler=data_sampler, num_workers=8, drop_last=True)

        data_loaders[name] = data_loader
        data_sizes[name] = data_sampler.__len__()
    return data_loaders, data_sizes

def hinge_loss(outputs, labels):
    num_labels = len(labels)
    correct_outputs = outputs[range(num_labels), labels]
    correct_outputs = correct_outputs.unsqueeze(0).T

    margin = 1.0
    margins = outputs - correct_outputs + margin
    max_margins, _ = torch.max(margins, dim=1)
    loss = torch.sum(max_margins) / len(labels)

    return loss

def train_model(
    data_loaders, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None
):
    # 时间
    since = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, dim=1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                lr_scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model

        # torch.save(best_model_weights, "./models/best_linear_svm_alexnet_car.pth")

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_loaders, data_sizes = load_data('D:\\Dataset\\PASCAL_VOC_2007\\finetune_car\\')

    # model_path = "./models/alexnet_car.pth"
    model = alexnet(pretrained=True)
    num_classes = 2
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    # model.load_state_dict(torch.load(model_path))
    # model.eval()

    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # model.classifier[6] = nn.Linear(num_features, num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model = train_model(data_loaders, model, criterion, optimizer, lr_scheduler, device=device, num_epochs=25)
    # 保存最好的模型参数
    check_dir('./models')
    torch.save(best_model.state_dict(), 'models/alexnet_car.pth')

    # hinge_loss_fn = hinge_loss

    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    #
    # train_model(
    #     data_loaders=data_loaders,
    #     model=model,
    #     criterion=hinge_loss_fn,
    #     optimizer=optimizer,
    #     lr_scheduler=lr_scheduler,
    #     num_epochs=10,
    #     device=device,
    # )
