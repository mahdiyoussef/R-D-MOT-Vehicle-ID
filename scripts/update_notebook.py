import json
import os
import re

nb_path = '/home/youssef/Desktop/vehicles-tracking-id/notebooks/dinov2_lora_veri776_kaggle.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

# Find Cell 2 and replace the XML parsing logic
new_dataset_source = """# ── Dataset Definition ────────────────────────────────────────────────────────
class VeRiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Determine image directory and annotations
        self.img_dir = os.path.join(root_dir, 'image_train')
        if not os.path.exists(self.img_dir):
            self.img_dir = root_dir # Fallback
            
        xml_path = os.path.join(root_dir, 'train_label.xml')
        type_txt_path = os.path.join(root_dir, 'list_type.txt')
        
        # Support if root_dir was pointed to image_train directly
        if not os.path.exists(xml_path):
            xml_path = os.path.join(os.path.dirname(root_dir), 'train_label.xml')
            type_txt_path = os.path.join(os.path.dirname(root_dir), 'list_type.txt')
            
        self.samples = []
        self.pids = set()
        
        # Load list_type.txt
        self.type_map = {}
        if os.path.exists(type_txt_path):
            with open(type_txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.type_map[int(parts[0])] = parts[1]
                        
        # Parse XML
        if os.path.exists(xml_path):
            print(f"Parsing annotations from {xml_path}")
            # VeRi uses gb2312 encoding which Python's ET fails to parse directly.
            # We read as string, strip the XML declaration, and parse.
            with open(xml_path, 'r', encoding='gb2312', errors='ignore') as f:
                xml_content = f.read()
            import re
            xml_content = re.sub(r"<\?xml.*?\?>", "", xml_content)
            
            root = ET.fromstring(xml_content)
            items = root.find('Items')
            if items is not None:
                for item in items.findall('Item'):
                    img_name = item.get('imageName')
                    pid = int(item.get('vehicleID'))
                    cam = item.get('cameraID')
                    color = int(item.get('colorID'))
                    type_id = int(item.get('typeID'))
                    
                    img_path = os.path.join(self.img_dir, img_name)
                    if os.path.exists(img_path):
                        self.samples.append({
                            'path': img_path,
                            'pid': pid,
                            'cam': cam,
                            'color': color,
                            'type_id': type_id,
                            'type_name': self.type_map.get(type_id, "unknown")
                        })
                        self.pids.add(pid)
        else:
            print(f"XML not found at {xml_path}, falling back to filename parsing.")
            self.image_paths = glob(os.path.join(self.img_dir, '*.jpg'))
            for path in self.image_paths:
                basename = os.path.basename(path)
                parts = basename.split('_')
                if len(parts) >= 2:
                    pid = int(parts[0])
                    self.samples.append({'path': path, 'pid': pid})
                    self.pids.add(pid)
                
        # Remap PIDs to 0..N-1
        self.pid2label = {pid: i for i, pid in enumerate(sorted(list(self.pids)))}
        self.num_classes = len(self.pids)
        print(f"Found {len(self.samples)} images across {self.num_classes} identities.")
        
        if self.type_map and os.path.exists(xml_path):
            type_counts = {}
            for s in self.samples:
                t = s.get('type_name', 'unknown')
                type_counts[t] = type_counts.get(t, 0) + 1
            print(f"Vehicle Types Distribution: {type_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample['path']
        pid = sample['pid']
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
# The dataset root should contain 'image_train/', 'train_label.xml', 'list_type.txt'
DATASET_PATH = "/kaggle/input/datasets/abhyudaya12/veri-vehicle-re-identification-dataset/VeRi" 
if not os.path.exists(DATASET_PATH):
    # Try another common path
    DATASET_PATH = "/kaggle/input/veri776/VeRi"
    
if os.path.exists(DATASET_PATH) or os.path.exists(os.path.join(DATASET_PATH, 'image_train')):
    train_dataset = VeRiDataset(DATASET_PATH, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
else:
    print(f"Dataset not found at {DATASET_PATH}! Please attach the dataset.")"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'class VeRiDataset(Dataset):' in source:
            lines = new_dataset_source.split('\n')
            cell['source'] = [line + ('\n' if i < len(lines)-1 else '') for i, line in enumerate(lines)]
            break

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook XML encoding issue fixed.")
