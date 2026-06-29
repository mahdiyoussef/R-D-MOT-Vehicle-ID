import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

md_intro = """# DINOv2 LoRA Fine-Tuning for Vehicle ReID (VeRi-776)
**Environment:** Kaggle Notebook (Dual T4 GPUs)
**Dataset:** VeRi-776. Please ensure you add a VeRi dataset to your Kaggle notebook (e.g. search "VeRi-776" or "veri dataset" in Kaggle Datasets and attach it). Typical path is `/kaggle/input/veri-dataset/VeRi`.

This notebook fine-tunes Meta's DINOv2 using Low-Rank Adaptation (LoRA) specifically for Vehicle Re-Identification. It leverages `torch.nn.DataParallel` to utilize both T4 GPUs."""

code_setup = """!pip install -q timm tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob
from tqdm.auto import tqdm

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))"""

code_dataset = """# ── Dataset Definition ────────────────────────────────────────────────────────
class VeRiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # VeRi image names: {vehicleID}_c{cameraID}_{frameID}_{...}.jpg
        # Example: 0002_c002_00030600_0.jpg
        self.image_paths = glob(os.path.join(root_dir, '*.jpg'))
        
        self.samples = []
        self.pids = set()
        
        for path in self.image_paths:
            basename = os.path.basename(path)
            parts = basename.split('_')
            if len(parts) >= 2:
                pid = int(parts[0])
                self.samples.append((path, pid))
                self.pids.add(pid)
                
        # Remap PIDs to 0..N-1
        self.pid2label = {pid: i for i, pid in enumerate(sorted(list(self.pids)))}
        self.num_classes = len(self.pids)
        print(f"Found {len(self.samples)} images across {self.num_classes} identities.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pid = self.samples[idx]
        label = self.pid2label[pid]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# Data Augmentation
transform_train = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# IMPORTANT: Update this path to match your attached Kaggle dataset
DATASET_PATH = "/kaggle/input/veri-dataset/image_train" 
if not os.path.exists(DATASET_PATH):
    # Try another common path
    DATASET_PATH = "/kaggle/input/veri776/VeRi/image_train"
    
if os.path.exists(DATASET_PATH):
    train_dataset = VeRiDataset(DATASET_PATH, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
else:
    print(f"Dataset not found at {DATASET_PATH}! Please attach the dataset.")"""

code_model = """# ── DINOv2 + LoRA Architecture ───────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, original, rank=8, alpha=32.0, dropout=0.1):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        d_in, d_out = original.in_features, original.out_features
        for p in self.original.parameters():
            p.requires_grad = False
        self.lora_A = nn.Linear(d_in, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_out, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

def inject_lora(model, target_modules=["qkv"], rank=8, alpha=32.0):
    targets = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                if any(t in f"{name}.{child_name}" for t in target_modules):
                    targets.append((module, child_name, child))
    for module, child_name, child in targets:
        setattr(module, child_name, LoRALinear(child, rank, alpha))
    return model

class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)

class DINOv2ReID(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load Frozen DINOv2
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Inject LoRA
        self.backbone = inject_lora(self.backbone, rank=8, alpha=32.0)
        
        # Heads
        self.gem = GeMPooling(p=3.0)
        embed_dim = 384 * 2 # CLS (384) + GeM (384)
        
        # BNNeck Projection
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        self.bottleneck.bias.requires_grad_(False)
        
        # Classifier for ID Loss
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]
        gem_feat = self.gem(patch_tokens)
        
        global_feat = torch.cat([cls_token, gem_feat], dim=1) # (B, 768)
        feat = self.bottleneck(global_feat)
        
        if self.training:
            cls_score = self.classifier(feat)
            return global_feat, cls_score
        return F.normalize(feat, p=2, dim=1)

# Initialize Model
if 'train_dataset' in locals():
    model = DINOv2ReID(num_classes=train_dataset.num_classes)
    
    # ── MULTI-GPU (T4x2) SETUP ──────────────────────────────────────────────────
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.2f}%)")"""

code_loss = """# ── Losses & Optimizer ──────────────────────────────────────────────────────
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

criterion_cls = nn.CrossEntropyLoss()
criterion_tri = TripletLoss(margin=0.3)

if 'train_dataset' in locals():
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)"""

code_train = """# ── Training Loop ───────────────────────────────────────────────────────────
EPOCHS = 10
SAVE_PATH = "/kaggle/working/dinov2_lora_veri776.pth"

if 'train_dataset' in locals():
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            global_feat, cls_score = model(imgs)
            
            loss_cls = criterion_cls(cls_score, labels)
            loss_tri = criterion_tri(global_feat, labels)
            loss = loss_cls + loss_tri
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(train_loader):.4f}")

    # ── Save Weights for inference ──────────────────────────────────────────────
    print("Training Complete! Saving weights...")
    
    # Extract underlying model from DataParallel
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    
    # Extract ONLY the LoRA matrices, GeM, and Bottleneck
    state_dict = model_to_save.state_dict()
    save_dict = {
        "backbone": {k: v for k, v in state_dict.items() if "lora_" in k},
        "gem_pool": {k.replace("gem.", ""): v for k, v in state_dict.items() if "gem." in k},
        "head": {k.replace("bottleneck.", "proj."): v for k, v in state_dict.items() if "bottleneck." in k}
    }
    
    torch.save(save_dict, SAVE_PATH)
    print(f"Weights successfully saved to {SAVE_PATH}!")
    print("Download this file from the Kaggle Data panel and place it in your local models/reid/ folder.")"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(md_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_code_cell(code_dataset),
    nbf.v4.new_code_cell(code_model),
    nbf.v4.new_code_cell(code_loss),
    nbf.v4.new_code_cell(code_train)
]

with open('notebooks/dinov2_lora_veri776_kaggle.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Jupyter notebook created successfully!")
