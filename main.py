from torch.utils.data import DataLoader
from clients.python import DatasetFS

dataset = DatasetFS()

loader = DataLoader(dataset, batch_size=None, num_workers=0)

for idx, batch in enumerate(loader):
    print(f"Прилетел Батч {idx}: shape {batch['image'].shape}")

    # model(batch["image"])
