import torch
import gradio as gr
from PIL import Image
from torch import nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from src.config import SELECTED_CLASSES
from src.transforms import get_baseline_transforms


#  MODEL_PATH = "models/augmented_model.pt"
MODEL_PATH = "models/baseline_model.pt"

# recreate the same model used during training 
def build_model(num_classes):
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def load_model():
    # loads trained weights into the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(num_classes=len(SELECTED_CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()   # put model in inference model 
    return model, device


model, device = load_model()
# apply resize, tensor conversion, and normalization to match training format
transform = get_baseline_transforms()   

# this function is called by the app when a user uploads an image 
# returns class name and its probability 
def predict(image):
    if image is None:
        return {}

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    return {
        SELECTED_CLASSES[i]: float(probabilities[i])
        for i in range(len(SELECTED_CLASSES))
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a document image"),
    outputs=gr.Label(num_top_classes=3, label="Predicted document type"),
    title="Robust Document Classifier",
    description=(
        "Upload a document image. The model uses transfer learning with MobileNetV2 "
        "to classify document types such as letters, forms, emails, handwritten notes, "
        "advertisements, and invoices. The augmented model was trained with realistic "
        "distortions such as blur, rotation, perspective shift, and contrast changes."
    ),
    examples=None,
)


if __name__ == "__main__":
    demo.launch()