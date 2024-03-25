import os
import torch
import torch.nn as nn
import argparse

import numpy as np
from utils.dataset import AVADataset
from utils.loss import emd_loss, dis_2_score, SupCRLoss
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score
import wandb
import random
from transformers import CLIPVisionModel
import datetime

os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
arg.add_argument("-n", "--task_name", required=False, default="CLIP-3407-bs64-0.1new_scr", type=str, help="task name")
arg.add_argument("--epochs", type=int, default=20)
arg.add_argument("-d", "--image_dir", required=False, default="D:\\Dataset\\AVA\\images", help="image dir")
arg.add_argument("-c", "--csv_dir", required=False, default="D:\\Dataset\\AVA\\labels", help="csv dir")
arg.add_argument("-w", "--use_wandb", required=False, type=int, default=1, help="use wandb or not")
opt = vars(arg.parse_args())

sweep_config = {
    'method': 'random'
}

metric = {
    'name': 'Accuracy',
    'goal': 'maximize'
}

sweep_config['metric'] = metric

sweep_config['parameters'] = {}
sweep_config['parameters'].update({
    'epochs': {'value': opt['epochs']},
    'ckpt_path': {'value': 'saved_models/swept_best_clip.pth'},
    'image_dir': {'value': opt['image_dir']},
    'csv_dir': {'value': opt['csv_dir']},
    'num_workers': {'value': 8},

})

# 离散型分布超参
sweep_config['parameters'].update({
    'optim_type': {
        'values': ['SGD']
    },
    'batch_size': {
        'value': 64
    },
})
# 连续型分布超参
sweep_config['parameters'].update({

    'lr': {
        'distribution': 'log_uniform_values',
        'min': 1e-5,
        'max': 1e-2,
    },
    'dropout_p': {
        'distribution': 'q_uniform',
        'q': 0.05,
        'min': 0.25,
        'max': 0.75,
    },
    'alpha': {
        'distribution': 'q_uniform',
        'q': 0.05,
        'min': 0.1,
        'max': 1.0,
    }
})

sweep_config['early_terminate'] = {
    'type': 'hyperband',
    'min_iter': 2,
    'eta': 2,
    's': 4
}


class AesClip(nn.Module):
    def __init__(self, dropout=0.1):
        super(AesClip, self).__init__()
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(768, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.clip(x)
        self.feature = x.pooler_output
        return self.proj(x.pooler_output)


def main():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project='Clip-sweep', config=wandb.config.__dict__, name=nowtime, save_code=True)
    # Load Data
    train_dataset = AVADataset(csv_file=os.path.normpath(os.path.join(opt['csv_dir'], 'train_labels.csv')),
                               root_dir=os.path.normpath(opt['image_dir']),
                               imgsz=224,  device=device, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=wandb.config.batch_size, shuffle=True,
                              num_workers=wandb.config.num_workers)

    val_dataset = AVADataset(csv_file=os.path.normpath(os.path.join(opt['csv_dir'], 'val_labels.csv')),
                             root_dir=os.path.normpath(opt['image_dir']),
                             imgsz=224, device=device, train=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=wandb.config.batch_size, shuffle=True,
                            num_workers=wandb.config.num_workers)
    teacher = AesClip(wandb.config.dropout_p)
    teacher = teacher.to(device)
    student = torchvision.models.resnet101(pretrained=True)
    student.fc = nn.Sequential(
        nn.Dropout(p=wandb.config.dropout_p),
        nn.Linear(student.fc.in_features, 10),
        nn.Softmax(dim=1)
    )
    student = student.to(device)
    criterion = emd_loss()
    # scl = SupCRLoss(temperature=wandb.config.temp)

    optimizer = torch.optim.__dict__[wandb.config.optim_type](
        student.parameters(), lr=wandb.config.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.1)

    best_acc = 0.0
    early_stop = 0
    teacher.eval()  # 将师生网络设置为评估模式
    # Train Network

    for epoch in range(wandb.config.epochs):
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

                loss = wandb.config.alpha * loss_hard + (1 - wandb.config.alpha) * loss_soft

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

            print(f"Epoch [{epoch + 1}/{wandb.config.epochs}], "
                  f"Validation Loss: {val_loss_r2:.4f}, "
                  f"MSE Loss: {mse_loss:.4f}, "
                  f"Pearson: {pearson:.4f}, "
                  f"Spearman: {spearman:.4f}, "
                  f"Accuracy: {acc:.4f}")
            if opt['use_wandb']:
                wandb.log({"val loss": val_loss_r2, "MSE loss": mse_loss, "Pearson": pearson, "Spearman": spearman,
                           "Accuracy": acc})

        lr_scheduler.step()


if __name__ == "__main__":
    from pprint import pprint

    pprint(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project='TS-sweep')
    wandb.agent(sweep_id=sweep_id, function=main, count=100)
