import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# DogDataset class
class DogDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.labels = []

        self.class_to_idx = {"dogs": 1, "notdogs": 0}
        for class_name in os.listdir(img_dir):
            class_path = os.path.join(img_dir, class_name)
            if not os.path.isdir(class_path) or class_name not in self.class_to_idx:
                continue

            for img_name in os.listdir(class_path):
                if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    continue
                self.images.append(os.path.join(class_path, img_name))
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
        return img, int(label)

# transformers and dataloaders
def get_transforms(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    img_size=224
) -> transforms.Compose:
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform

def get_train_loader(
    img_dir,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
):
    transform = get_transforms()
    train_ds = DogDataset(img_dir=img_dir, transform=transform)
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return train_loader

def get_test_loader(
    img_dir,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
):
    transform = get_transforms()
    test_ds = DogDataset(img_dir=img_dir, transform=transform)
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return test_loader