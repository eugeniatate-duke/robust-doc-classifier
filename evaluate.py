import argparse
import json

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch import nn

from src.config import SELECTED_CLASSES
from src.data import RVLCDIPSubset
from src.transforms import get_baseline_transforms

# training models can be misleading as models can memorize training data
# this file helps demonstrate how the model generalizes and compares baseline and augmented models 

def build_model(num_classes):
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def evaluate_model(model_path, max_per_class):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load test dataset (unseen data)
    dataset = RVLCDIPSubset(
        split="test",
        transform=get_baseline_transforms(),   # baseline transform (keep test data standardized)
        max_per_class=max_per_class,
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    # rebuilds same model used during training 
    model = build_model(num_classes=len(SELECTED_CLASSES))
    # load trained weights from baseline or augmented model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    # inference: images pass through and predicted classes are output 
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    # compute accuracy 
    accuracy = accuracy_score(y_true, y_pred)
    # create classification report: recall, precision, F1 score per class 
    report = classification_report(
        y_true,
        y_pred,
        target_names=SELECTED_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    # build confusion matrix 
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, report, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-per-class", type=int, default=50)
    args = parser.parse_args()

    accuracy, report, cm = evaluate_model(args.model_path, args.max_per_class)

    results = {
        "model_name": args.name,
        "accuracy": accuracy,
        "classification_report": report,
    }

    with open(f"figures/{args.name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=SELECTED_CLASSES,
    )
    disp.plot(xticks_rotation=45)
    plt.title(f"{args.name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"figures/{args.name}_confusion_matrix.png")
    plt.close()

    print(f"{args.name} accuracy: {accuracy:.4f}")
    print(f"Saved results to figures/{args.name}_results.json")
    print(f"Saved confusion matrix to figures/{args.name}_confusion_matrix.png")


if __name__ == "__main__":
    main()