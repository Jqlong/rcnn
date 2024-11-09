import copy
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import alexnet

from utils.custom_classifier_dataset import CustomClassifierDataset


def load_data(data_root_dir):
    # データの前処理を定義する
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((227, 227)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # データを格納する辞書を用意する
    data_loaders = {}
    data_sizes = {}
    remain_negative_list = list()
    # 学習データと検証データを取得する
    for name in ["train", "val"]:
        data_dir = os.path.join(data_root_dir, name)

        # データをロードする
        data_set = CustomClassifierDataset(data_dir, transform=transform)
        # 学習データの場合は、負のサンプルをランダムに選択する
        if name is "train":
            positive_list = data_set.get_positives()
            negative_list = data_set.get_negatives()

            init_negative_idxs = random.sample(
                range(len(negative_list)), len(positive_list)
            )
            init_negative_list = [
                negative_list[idx]
                for idx in range(len(negative_list))
                if idx in init_negative_idxs
            ]
            remain_negative_list = [
                negative_list[idx]
                for idx in range(len(negative_list))
                if idx not in init_negative_idxs
            ]

            data_set.set_negative_list(init_negative_list)
            data_loaders["remain"] = remain_negative_list

        # データローダーを作成する
        data_loader = DataLoader(
            data_set,
            batch_size=len(data_set),
            num_workers=8,
            drop_last=True,
        )
        data_loaders[name] = data_loader
        data_sizes[name] = len(data_set)
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
    best_acc = 0.0

    for epoch in range(num_epochs):
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

        torch.save(best_model_weights, "./models/best_linear_svm_alexnet_car.pth")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_loaders, data_sizes = load_data("./data/classifier_car")

model_path = "./models/alexnet_car.pth"
model = alexnet()
num_classes = 2
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

for param in model.parameters():
    param.requires_grad = False

model.classifier[6] = nn.Linear(num_features, num_classes)

model = model.to(device)

hinge_loss_fn = hinge_loss

optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

train_model(
    data_loaders=data_loaders,
    model=model,
    criterion=hinge_loss_fn,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    num_epochs=10,
    device=device,
)