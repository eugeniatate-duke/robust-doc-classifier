from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image

from src.config import ORIGINAL_ID_TO_NEW_ID

# handles loading and filtering of the dataset; feeds batches of images into the model upon training 
class RVLCDIPSubset(Dataset):
    def __init__(self, split="train", transform=None, max_per_class=300):
        # self.dataset = load_dataset("aharley/rvl_cdip", split=split)
        self.dataset = load_dataset("dvgodoy/rvl_cdip_mini", split=split)
        self.transform = transform
        self.samples = []

        class_counts = {}

        for i, item in enumerate(self.dataset):
            label = item["label"]

            # keep only pre-selected classes
            if label not in ORIGINAL_ID_TO_NEW_ID:
                continue

            new_label = ORIGINAL_ID_TO_NEW_ID[label]
            class_counts[new_label] = class_counts.get(new_label, 0)

            # limit number of images per class
            if class_counts[new_label] >= max_per_class:
                continue

            self.samples.append(i)
            class_counts[new_label] += 1

            if len(class_counts) == len(ORIGINAL_ID_TO_NEW_ID):
                if all(count >= max_per_class for count in class_counts.values()):
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.dataset[self.samples[idx]]
        # convert image to RGB
        image = item["image"].convert("RGB")
        label = ORIGINAL_ID_TO_NEW_ID[item["label"]]

        # apply transform 
        if self.transform:
            image = self.transform(image)

        # return image-label pairs
        return image, label