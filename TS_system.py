import os
import torch
import torch.nn as nn
import torchvision
import argparse

import numpy as np
from vit_pytorch import ViT
from utils.dataset import AVADataset
from utils.loss import emd_loss, dis_2_score, SupCon, SupCRLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score
import wandb
import random
from transformers import CLIPVisionModel, CLIPVisionConfig, AutoProcessor
from train_clip import AesClip

arg = argparse.ArgumentParser()
arg.add_argument("-n", "--task_name", required=False, default="Teacher-Student-a0.7", type=str, help="task name")
arg.add_argument("--batch_size", type=int, default=64)
arg.add_argument("--epoch", type=int, default=20)
arg.add_argument("--lr", type=float, default=1e-1)
arg.add_argument("-d", "--image_dir", required=False, default="D:\\Dataset\\AVA\\images", help="image dir")
arg.add_argument("-c", "--csv_dir", required=False, default="D:\\Dataset\\AVA\\labels", help="csv dir")
arg.add_argument("-w", "--use_wandb", required=False, type=int, default=1, help="use wandb or not")
arg.add_argument("-a", "--alpha", required=False, type=float, default=0.7, help="control the weight of ts loss")
opt = vars(arg.parse_args())


num_classes = 10
num_epochs = opt['epoch']
batch_size = opt['batch_size']
LR = opt['lr']
imgsz = 224
num_workers = 8 if opt['use_wandb'] else 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
# 加载数据集（这里使用MNIST作为示例）
    train_dataset = AVADataset(csv_file=os.path.normpath(os.path.join(opt['csv_dir'], 'train_labels.csv')),
                               root_dir=os.path.normpath(opt['image_dir']), imgsz=imgsz, device=device, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = AVADataset(csv_file=os.path.normpath(os.path.join(opt['csv_dir'], 'val_labels.csv')),
                             root_dir=os.path.normpath(opt['image_dir']),
                             imgsz=imgsz, device=device, train=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 初始化师生模型、损失函数和优化器
    teacher = AesClip()
    teacher = teacher.to(device)
    student = torchvision.models.resnet101(pretrained=False)
    student.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(student.fc.in_features, num_classes),
            nn.Softmax(dim=1)
        )
    student = student.to(device)


    criterion = emd_loss()
    optimizer = torch.optim.SGD(student.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    ls_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.1)


    best_acc = 0
    early_stop = 0
    teacher.eval()  # 将师生网络设置为评估模式

    for epoch in range(num_epochs):
        student.train()  # 将学生网络设置为训练模式
        torch.cuda.empty_cache()
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                optimizer.zero_grad()
                inputs, labels = data['image'], data['annotations']
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.no_grad():
                    teacher_output = teacher(inputs)
                student_output = student(inputs)

                loss_hard = criterion(student_output, labels)
                loss_soft = criterion(student_output, teacher_output)


                loss = opt['alpha']* loss_hard + (1-opt['alpha']) * loss_soft

                # Backward

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())
                if i % 100 == 0 and opt['use_wandb']:
                    wandb.log({"train loss": loss.item()})

        student.eval()
        with torch.no_grad():
            val_loss_r2 = []
            pred_list = []
            target_list = []
            pred_score_list = []
            target_score_list = []
            for data in tqdm(val_loader, unit="batch"):
                inputs, labels = data['image'], data['annotations']
                inputs, labels = inputs.to(device), labels.to(device)


                outputs = student(inputs)

                loss = criterion(outputs, labels)
                val_loss_r2.append(loss.item())
                pred_list.append(outputs)
                target_list.append(labels)
                pred_score_list += dis_2_score(outputs).tolist()
                target_score_list += dis_2_score(labels).tolist()

            val_loss_r2 = sum(val_loss_r2) / len(val_loss_r2)
            mse_loss = torch.nn.functional.mse_loss(torch.tensor(pred_score_list),
                                                    torch.tensor(target_score_list)).item()

            # 计算皮尔逊相关系数
            pearson = pearsonr(pred_score_list, target_score_list)[0]
            # 计算斯皮尔曼相关系数
            spearman = spearmanr(pred_score_list, target_score_list)[0]

            pred_score_list = np.array(pred_score_list)
            target_score_list = np.array(target_score_list)

            pred_label = np.where(pred_score_list <= 5.00, 0, 1)
            target_label = np.where(target_score_list <= 5.00, 0, 1)

            acc = accuracy_score(target_label, pred_label)

            if acc > best_acc:
                best_acc = acc
                torch.save(student.state_dict(), os.path.join("saved_models", f"student_model_{best_acc}.pth"))
                print(f"Best model saved, acc: {acc}")
                early_stop = 0
            else:
                early_stop += 1
                if early_stop > 5:
                    print("Early stop")
                    break

            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Validation Loss: {val_loss_r2:.4f}, "
                  f"MSE Loss: {mse_loss:.4f}, "
                  f"Pearson: {pearson:.4f}, "
                  f"Spearman: {spearman:.4f}, "
                  f"Accuracy: {acc:.4f}")
            if opt['use_wandb']:
                wandb.log({"val loss": val_loss_r2, "MSE loss": mse_loss, "Pearson": pearson, "Spearman": spearman,
                           "Accuracy": acc})

        ls_scheduler.step()


if __name__ == "__main__":
    if opt['use_wandb']:
        wandb.init(project="AVANew", name=opt['task_name'],
                   config={"batch_size": batch_size, "num_epochs": num_epochs})
    main()

