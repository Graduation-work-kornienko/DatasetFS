from torch.utils.data import DataLoader
from clients.python import DatasetFS

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import matplotlib.pyplot as plt

import os
import webdataset as wds

DEVICE = "cpu"
BATCH_SIZE=32
EPOCHS=10

class SimpleCNN(nn.Module):
    """simple model for verification"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def fs_collate_fn(batch):
    if not batch:
        return {}

    TARGET_SIZE = (224, 224)

    images = []
    targets = []
    meta_list = []

    for item in batch:
        img = item['image']


        img_resized = F.interpolate(
            img.unsqueeze(0),
            size=TARGET_SIZE,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        images.append(img_resized)


        label = 0
        if 'annotations' in item and len(item['annotations']) > 0:
            label = item['annotations'][0].get('category_id', 0)
        elif 'category_id' in item:
            label = item['category_id']

        targets.append(torch.tensor(label, dtype=torch.long))

        meta_list.append(item)

    return {
        'images': torch.stack(images),
        'targets': torch.stack(targets),
        'meta': meta_list
    }


dataset = DatasetFS(
    shm_data_path="/tmp/mlfs_data.bin",
    shm_refs_path="/tmp/mlfs_refs.bin",
    pipe_path="/tmp/datasetfs_pipe"
)

loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=fs_collate_fn,
    num_workers=0,
    pin_memory=True
)


def get_fs_loader():
    """Загрузчик для вашей файловой системы (SHM + Pipe)"""
    print("[FS] Инициализация DatasetFS...")
    dataset = DatasetFS(
        shm_data_path="/tmp/mlfs_data.bin",
        shm_refs_path="/tmp/mlfs_refs.bin",
        pipe_path="/tmp/datasetfs_pipe",
        # cache_entire_epoch=True,
        timeout_seconds=10.0
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=fs_collate_fn,
        num_workers=0,
        pin_memory=True,
        shuffle=False
    )

def get_wds_loader(pattern):
    """Загрузчик для WebDataset из реальных файлов"""
    if not os.path.exists(pattern.replace("{000000..000001}.tar", "").rstrip("/")):
        print(f"[WDS] Внимание: Путь {pattern} может быть неверным. Проверьте файлы.")

    print(f"[WDS] Чтение шардов: {pattern}")

    dataset = (
        wds.WebDataset(pattern)
        .shuffle(1000)
        .decode('rgb')
        .map(lambda sample: {"image": sample[0], "json": sample[1]})
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=fs_collate_fn,
        num_workers=0,
        pin_memory=True
    )



def train_one_epoch(model, loader, criterion, optimizer, device, name):
    model.train()
    total_loss = 0.0
    count = 0

    for batch_idx, batch in enumerate(loader):
        if not batch or 'images' not in batch:
            continue

        images = batch['images'].to(device)
        targets = batch['targets'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

        # Логирование каждые 10 батчей
        if batch_idx % 10 == 0:
            print(f"[{name}] Epoch Batch {batch_idx}, Loss: {loss.item():.4f}, Img Shape: {images.shape}")

    return total_loss / max(count, 1)

def run_comparison():
    print(f"Используемое устройство: {DEVICE}")

    losses_fs = []
    losses_wds = []

    # Dataset FS
    print("\n" + "="*50)
    print("ЗАПУСК 1: DatasetFS (SHM+Pipe)")
    print("="*50)

    try:
        loader_fs = get_fs_loader()
        model_fs = SimpleCNN().to(DEVICE)
        optimizer_fs = optim.Adam(model_fs.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{EPOCHS} (Custom FS) ---")
            loss = train_one_epoch(model_fs, loader_fs, criterion, optimizer_fs, DEVICE, "FS")
            losses_fs.append(loss)
            print(f"[FS] Epoch {epoch+1} finished. Avg Loss: {loss:.4f}")

    except Exception as e:
        print(f"[FS] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        losses_fs = [None] * EPOCHS

    # WebDataset
    print("\n" + "="*50)
    print("ЗАПУСК 2: WebDataset (Real Shards)")
    print("="*50)

    try:
        if not os.path.exists(WDS_PATTERN.split('{')[0]):
             pass

        loader_wds = get_wds_loader(WDS_PATTERN)

        model_wds = SimpleCNN().to(DEVICE)

        optimizer_wds = optim.Adam(model_wds.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{EPOCHS} (WebDataset) ---")
            loss = train_one_epoch(model_wds, loader_wds, criterion, optimizer_wds, DEVICE, "WDS")
            losses_wds.append(loss)
            print(f"[WDS] Epoch {epoch+1} finished. Avg Loss: {loss:.4f}")

    except Exception as e:
        print(f"[WDS] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        losses_wds = [None] * EPOCHS

    # Visualization
    plt.figure(figsize=(12, 7))

    valid_fs = [x for x in losses_fs if x is not None]
    valid_wds = [x for x in losses_wds if x is not None]

    if valid_fs:
        plt.plot(valid_fs, label='DatasetFS', marker='o', linewidth=2, markersize=8)
    if valid_wds:
        plt.plot(valid_wds, label='WebDataset', marker='x', linestyle='--', linewidth=2, markersize=8)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Сравнение обучения: Dataset FS vs WebDataset', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(max(valid_fs, valid_wds, key=len))))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    WDS_PATTERN = "cmd/dataset_converter/publaynet-train-{000000..000001}.tar"

    import glob
    files = glob.glob(WDS_PATTERN.replace("{000000..000001}.tar", "*.tar"))
    if not files:
            print("Не удалось найти файлы по паттерну автоматически. Убедитесь, что WDS_PATTERN верен.")
    else:
        print(f"Найдено шардов WebDataset: {len(files)}")

    run_comparison()

# for idx, batch in enumerate(loader):
#     print(f"Прилетел Батч {idx}: shape {batch['image'].shape}")

    # model(batch["image"])
