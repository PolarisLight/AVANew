import os
import torch
import argparse
import datetime

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
arg.add_argument("-n", "--task_name", required=False, default="MPCLIP", type=str, help="task name")
arg.add_argument("--epochs", type=int, default=40)
arg.add_argument("-d", "--image_dir", required=False, default="D:\\Dataset\\AVA\\images", help="image dir")
arg.add_argument("-c", "--csv_dir", required=False, default="D:\\Dataset\\AVA\\labels", help="csv dir")
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
        'values': ['AdamW']
    },
    'batch_size': {
        'value': 64
    },
})
# 连续型分布超参
sweep_config['parameters'].update({

    'lr_clip': {
        'distribution': 'log_uniform_values',
        'min': 1e-7,
        'max': 1e-5,
    },
    'lr_proj': {
        'distribution': 'log_uniform_values',
        'min': 1e-5,
        'max': 1e-3,
    },
    'dropout_p': {
        'distribution': 'q_uniform',
        'q': 0.05,
        'min': 0.25,
        'max': 0.75,
    },
    'patch_num': {
        'distribution': 'q_uniform',
        'q': 1,
        'min': 1,
        'max': 10,
    },
    'beta': {
        'distribution': 'q_uniform',
        'q': 0.05,
        'min': 0.25,
        'max': 2.0,
    },
    'k': {
        'distribution': 'q_uniform',
        'q': 0.05,
        'min': 0.25,
        'max': 2.0,
    },
})

sweep_config['early_terminate'] = {
    'type': 'hyperband',
    'min_iter': 2,
    'eta': 2,
    's': 4
}


def main():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project='Clip-sweep', config=wandb.config.__dict__, name=nowtime, save_code=True)
    # Load Data
    train_dataset = AVADatasetMP(csv_file=os.path.normpath(os.path.join(opt['csv_dir'], 'train_labels.csv')),
                                 root_dir=os.path.normpath(opt['image_dir']),
                                 imgsz=224, device=device, train=True, patch_num=int(wandb.config.patch_num))
    train_loader = DataLoader(dataset=train_dataset, batch_size=wandb.config.batch_size, shuffle=True,
                              num_workers=wandb.config.num_workers)

    val_dataset = AVADatasetMP(csv_file=os.path.normpath(os.path.join(opt['csv_dir'], 'val_labels.csv')),
                               root_dir=os.path.normpath(opt['image_dir']),
                               imgsz=224, device=device, train=False, patch_num=int(wandb.config.patch_num))
    val_loader = DataLoader(dataset=val_dataset, batch_size=wandb.config.batch_size, shuffle=True,
                            num_workers=wandb.config.num_workers)
    model = AesClipMP().to(device)

    criterion = MPEMDLoss(norm=True, beta=wandb.config.beta, k=wandb.config.k)
    criterion_val = emd_loss()

    optimizer = torch.optim.__dict__[wandb.config.optim_type](
        [{"params": model.clip.parameters(), "lr": wandb.config.lr_clip},
         {"params": model.proj.parameters(), "lr": wandb.config.lr_proj}])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.1)

    best_acc = 0.0
    early_stop = 0

    # Train Network
    for epoch in range(wandb.config.epochs):
        model.train()
        torch.cuda.empty_cache()

        with tqdm(train_loader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                inputs, labels = data['image'], data['annotations']
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss_emd = criterion(outputs, labels)

                loss = loss_emd
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())
                if i % 100 == 0:
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
                torch.save(model.state_dict(), os.path.join("saved_models", f"mpclip_model_{best_acc}.pth"))
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

            wandb.log({"val loss": val_loss_r2, "MSE loss": mse_loss, "Pearson": pearson, "Spearman": spearman,
                       "Accuracy": acc})

        lr_scheduler.step()
        


if __name__ == "__main__":
    from pprint import pprint

    pprint(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project='MPClip-sweep')
    wandb.agent(sweep_id=sweep_id, function=main, count=100)
