import torch
import torchvision
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionTransformer
import math


class AesClip(nn.Module):
    def __init__(self):
        super(AesClip, self).__init__()
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.proj = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.clip(x)
        self.feature = x.pooler_output
        return self.proj(x.pooler_output)


class AesClipMP(nn.Module):
    def __init__(self):
        super(AesClipMP, self).__init__()
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.proj = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        patches = x.shape[1]
        x = x.reshape(-1, 3, 224, 224)
        x = self.clip(x)
        self.feature = x.pooler_output
        proj = self.proj(x.pooler_output)
        return proj.contiguous().view(-1, patches, 10)


if __name__ == '__main__':
    from dataset import AVADatasetMP
    from torch.utils.data import DataLoader
    dateset = AVADatasetMP(csv_file='D:\\Dataset\\AVA\\labels\\train_labels.csv',
                           root_dir='D:\\Dataset\\AVA\\images', imgsz=224, patch_num=10)
    dataloader = DataLoader(dataset=dateset, batch_size=32, shuffle=True, num_workers=0)
    config = CLIPVisionConfig(projection_dim=10)
    vit = AesClipMP()
    print(vit)
    data = next(iter(dataloader))
    print(data['image'].shape)
    print(torch.cuda.memory_allocated())
    vit = vit.to('cuda')
    print(f"{torch.cuda.memory_allocated():,}")
    data['image'] = data['image'].to('cuda')
    print(f"{torch.cuda.memory_allocated():,}")
    output = vit(data['image'])
    print(output.shape)
    print(f"{torch.cuda.max_memory_allocated()/1024/1024/1024:,} GB")

