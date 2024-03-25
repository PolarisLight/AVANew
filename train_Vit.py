import os
import torch
import torchvision
import argparse

import numpy as np
from utils.dataset import AVADataset
from utils.loss import emd_loss, dis_2_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTConfig, Swinv2ForImageClassification, Swinv2Config, \
    SwinForImageClassification, SwinConfig
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score
import wandb
import random

os.environ["TORCH_USE_CUDA_DSA"] = "1"

seed = 3407
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arg = argparse.ArgumentParser()
arg.add_argument("-n", "--task_name", required=False, default="Swin-3407-bs64-individualLR", type=str, help="task name")
arg.add_argument("--batch_size", type=int, default=64)
arg.add_argument("--epoch", type=int, default=20)
arg.add_argument("--lr_backbone", type=float, default=3e-6)
arg.add_argument("--lr_proj", type=float, default=3e-4)
arg.add_argument("-d", "--image_dir", required=False, default="D:\\Dataset\\AVA\\images", help="image dir")
arg.add_argument("-c", "--csv_dir", required=False, default="D:\\Dataset\\AVA\\labels", help="csv dir")
arg.add_argument("-w", "--use_wandb", required=False, type=int, default=1, help="use wandb or not")
opt = vars(arg.parse_args())

# Hyperparameters
num_classes = 10
num_epochs = opt['epoch']
batch_size = opt['batch_size']
LR_BACKBONE = opt['lr_backbone']
LR_PROJ = opt['lr_proj']
imgsz = 224
num_workers = 8 if opt['use_wandb'] else 0


def main():
    # Load Data
    train_dataset = AVADataset(csv_file=os.path.normpath(os.path.join(opt['csv_dir'], 'train_labels.csv')),
                               root_dir=os.path.normpath(opt['image_dir']), imgsz=imgsz, device=device, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = AVADataset(csv_file=os.path.normpath(os.path.join(opt['csv_dir'], 'val_labels.csv')),
                             root_dir=os.path.normpath(opt['image_dir']),
                             imgsz=imgsz, device=device, train=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model

    # model = ViT(
    #     image_size=imgsz,
    #     patch_size=32,
    #     num_classes=num_classes,
    #     dim=1024,
    #     depth=6,
    #     heads=16,
    #     mlp_dim=2048,
    #     dropout=0.1,
    #     emb_dropout=0.1
    # ).to(device)
    # summary(model, (3, imgsz, imgsz))
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params}")

    # config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
    # config.image_size = imgsz
    # config.num_labels = num_classes
    # # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    # model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', config=config,
    #                                                   ignore_mismatched_sizes=True).to(device)
    config = SwinConfig.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
    config.image_size = imgsz
    config.num_labels = num_classes
    model = SwinForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224', config=config,
                                                         ignore_mismatched_sizes=True).to(device)

    # model = torchvision.models.resnet101(pretrained=True)
    # model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    # model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # Loss and optimizer
    criterion = emd_loss()

    optimizer = torch.optim.Adam([{"params": model.swin.parameters(), "lr": LR_BACKBONE}
                                     , {"params": model.classifier.parameters(), "lr": LR_PROJ}
                                  ])
    ls_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.1)

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

                # Forward
                # outputs = model(inputs)
                # inputs = feature_extractor(images=inputs, return_tensors="pt")
                inputs = {"pixel_values": inputs}
                outputs = model(**inputs).logits

                # outputs = model(inputs)
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
                inputs = {"pixel_values": inputs}
                outputs = model(**inputs).logits

                # outputs = model(inputs)

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
                torch.save(model.state_dict(), os.path.join("saved_models", f"vit_model_{epoch}.pth"))
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
    save_path = 'saved_models'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + '/vit_model.pth')
    print('Model saved')


if __name__ == "__main__":
    if opt['use_wandb']:
        wandb.init(project="AVANew", name=opt['task_name'],
                   config={"batch_size": batch_size, "num_epochs": num_epochs})
    main()
