import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from tqdm import tqdm

from src.data import RVLCDIPSubset
from src.transforms import get_baseline_transforms, get_augmented_transforms
from src.config import SELECTED_CLASSES


def build_model(num_classes):
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    # freeze pretrained feature extractor
    for param in model.features.parameters():
        param.requires_grad = False
    # replcae final classification layer with a new layer for our doc classes 
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def main():

    # read cmd line args 
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmented", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-per-class", type=int, default=300)
    args = parser.parse_args()

    # selects CPU or GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # choose transforms 
    transform = get_augmented_transforms() if args.augmented else get_baseline_transforms()

    train_dataset = RVLCDIPSubset(
        split="train",
        transform=transform,
        max_per_class=args.max_per_class,
    )
    # loads a data subset 
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )

    # build a pretrained model
    model = build_model(num_classes=len(SELECTED_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    # train only classifier model: only update classifier weights and keep pretrained feature extractor unchanged 
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}: loss={loss:.4f}, accuracy={acc:.4f}")

    model_name = "augmented_model.pt" if args.augmented else "baseline_model.pt"
    torch.save(model.state_dict(), f"models/{model_name}")
    print(f"Saved model to models/{model_name}")


if __name__ == "__main__":
    main()