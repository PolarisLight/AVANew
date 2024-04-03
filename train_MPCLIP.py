import os
import torch
import argparse

import numpy as np
from utils.dataset import AVADatasetMP
from utils.loss import emd_loss, dis_2_score, MPEMDLoss
from utils.models import AesClipMP
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score
import wandb
import random
import platform

os.environ["TORCH_USE_CUDA_DSA"] = "1"

seed = 3407


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU


set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arg = argparse.ArgumentParser()
arg.add_argument("-n", "--task_name", required=False, default="MPCLIP-3407-bs64-SGD", type=str, help="task name")
arg.add_argument("--batch_size", type=int, default=64)
arg.add_argument("--epoch", type=int, default=40)
arg.add_argument("--lr_clip", type=float, default=3e-5)
arg.add_argument("--lr_proj", type=float, default=3e-3)
arg.add_argument("--patch_num", type=int, default=5)
arg.add_argument("-d", "--image_dir", required=False, default="D:\\Dataset\\AVA\\images", help="image dir")
arg.add_argument("-c", "--csv_dir", required=False, default="D:\\Dataset\\AVA\\labels", help="csv dir")
arg.add_argument("-w", "--use_wandb", required=False, type=int, default=1, help="use wandb or not")
opt = vars(arg.parse_args())

# Hyperparameters
num_classes = 10
num_epochs = opt['epoch']
batch_size = opt['batch_size']
LR_CLIP = opt['lr_clip']
LR_PROJ = opt['lr_proj']
imgsz = 224
num_workers = 8 if opt['use_wandb'] else 0


def main():
    # Load Data
    train_dataset = AVADatasetMP(csv_file=os.path.normpath(os.path.join(opt['csv_dir'], 'train_labels.csv')),
                                 root_dir=os.path.normpath(opt['image_dir']), imgsz=imgsz, device=device, train=True,
                                 patch_num=opt['patch_num'])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = AVADatasetMP(csv_file=os.path.normpath(os.path.join(opt['csv_dir'], 'val_labels.csv')),
                               root_dir=os.path.normpath(opt['image_dir']),
                               imgsz=imgsz, device=device, train=False, patch_num=opt['patch_num'])
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model
    # config = CLIPVisionConfig(projection_dim=num_classes)
    # model = CLIPVisionModel(config).cuda()
    model = AesClipMP().to(device)
    if platform.system() == "Linux":
        model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # Loss and optimizer
    criterion = MPEMDLoss()
    criterion_val = emd_loss()

    optimizer = torch.optim.SGD([{"params": model.clip.parameters(), "lr": LR_CLIP},
                                   {"params": model.proj.parameters(), "lr": LR_PROJ}])
    ls_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.7)

    best_acc = 0.0
    early_stop = 0

    # Train Network
    for epoch in range(num_epochs):
        model.train()
        torch.cuda.empty_cache()
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                inputs, labels = data['image'], data['annotations']
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
                if i % 100 == 0 and opt['use_wandb']:
                    wandb.log({"train loss": loss.item()})
        model.eval()
        with torch.no_grad():
            val_loss_r2 = []
            pred_list = []
            target_list = []
            pred_score_list = []
            target_score_list = []
            for data in tqdm(val_loader, unit="batch"):
                inputs, labels = data['image'], data['annotations']
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                outputs = torch.mean(outputs, dim=1)

                loss = criterion_val(outputs, labels)
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
                torch.save(model.state_dict(), os.path.join("saved_models", f"clip_model_{best_acc}.pth"))
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

    # Save model
    # save_path = 'saved_models'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # torch.save(model.state_dict(), save_path + '/clip_model.pth')
    # print('Model saved')


if __name__ == "__main__":
    if opt['use_wandb']:
        wandb.init(project="AVANew", name=opt['task_name'],
                   config={"batch_size": batch_size, "num_epochs": num_epochs})
    main()
