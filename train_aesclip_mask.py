import os
import torch
import torch.nn as nn
import torchvision
import argparse

import numpy as np
from utils.dataset import AVADatasetSAM_New
from utils.loss import emd_loss, dis_2_score
from torch.utils.data import DataLoader
import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score
import wandb
import random
from transformers import CLIPVisionModel
from utils.CustomCLIP import AesClipCA

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
arg.add_argument("-n", "--task_name", required=False, default="maskCLIP-3407-bs64-0.1new_scr", type=str,
                 help="task name")
arg.add_argument("--batch_size", type=int, default=64)
arg.add_argument("--epoch", type=int, default=20)
arg.add_argument("--lr_clip", type=float, default=3e-6)
arg.add_argument("--lr_proj", type=float, default=3e-4)
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


class AesClip(nn.Module):
    def __init__(self):
        super(AesClip, self).__init__()
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.proj = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.clip(x)
        self.feature = x.pooler_output
        return self.proj(x.pooler_output)



def train(model, train_loader, val_loader, criterion_train, criterion_test, optimizer, epochs=10,
          model_saved_path=None):
    """
    training function
    train the model and only save the better model on validation set in order to avoid storge problem
    :param model: model to be trained
    :param train_loader: data loader of training set
    :param val_loader: data loader of validation set
    :param criterion: loss function
    :param optimizer: optimizer
    :param epochs: total epochs
    :param model_saved_path: path to save the model, if None, the model will not be saved. default: None
    :return:
    """


def main():
    train_dataset = AVADatasetSAM_New(csv_file=os.path.normpath(os.path.join(opt['csv_dir'], 'train_labels.csv')),
                                      root_dir=os.path.normpath(opt['image_dir']),
                                      mask_num=50,
                                      imgsz=(imgsz, imgsz), if_test=False,
                                      transform=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    val_dataset = AVADatasetSAM_New(csv_file=os.path.normpath(os.path.join(opt['csv_dir'], 'val_labels.csv')),
                                                              root_dir=os.path.normpath(opt['image_dir']), mask_num=50,
                                                              imgsz=(imgsz, imgsz),
                                                              if_test=True,
                                                              transform=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model
    # config = CLIPVisionConfig(projection_dim=num_classes)
    # model = CLIPVisionModel(config).cuda()
    model = AesClipCA().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # Loss and optimizer
    criterion_train = emd_loss(dist_r=2)
    criterion_test = emd_loss(dist_r=1)

    optimizer = torch.optim.Adam([{"params": model.clip.parameters(), "lr": LR_CLIP},
                                  {"params": model.proj.parameters(), "lr": LR_PROJ}])
    ls_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.1)

    previous_val_loss = 1e10
    early_stop = 0
    best_acc = 0
    for epoch in range(num_epochs):

        if opt["use_wandb"]:
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
        model.train()
        with tqdm.tqdm(train_loader, unit='batch') as pbar:
            for batch_idx, datas in enumerate(train_loader):
                data, target, mask = datas["image"].to(device), datas["annotations"].to(device), datas["masks"].to(
                    device)
                mask_loc = datas["mask_loc"].to(device)
                optimizer.zero_grad()
                output = model(data, mask)
                loss = criterion_train(output, target)
                if not opt["use_wandb"]:
                    print()
                    print(f"0:output:{output[0].detach().cpu().numpy()}, "
                          f"\ntarget:{target[0].detach().cpu().numpy()}")
                loss.backward()

                optimizer.step()
                pbar.update(1)
                pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()
                ))

                if batch_idx % 100 == 0 and opt['use_wandb']:
                    wandb.log({"train loss": loss.item()})


        model.eval()
        with torch.no_grad():
            val_loss_r2 = []
            val_loss_r1 = []
            pred_list = []
            target_list = []
            pred_score_list = []
            target_score_list = []
            for datas in tqdm.tqdm(val_loader):
                data, target, mask = datas["image"].to(device), datas["annotations"].to(device), datas["masks"].to(
                    device)
                mask_loc = datas["mask_loc"].to(device)
                output = model(data, mask)
                val_loss_r2.append(criterion_train(output, target).item())
                val_loss_r1.append(criterion_test(output, target).item())
                pred_list.append(output)
                target_list.append(target)
                pred_score_list += dis_2_score(output).tolist()
                target_score_list += dis_2_score(target).tolist()

            val_loss_r2 = sum(val_loss_r2) / len(val_loss_r2)
            val_loss_r1 = sum(val_loss_r1) / len(val_loss_r1)

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

if __name__ == "__main__":
    if opt['use_wandb']:
        wandb.init(project="AVANew", name=opt['task_name'],
                   config={"batch_size": batch_size, "num_epochs": num_epochs})
    main()