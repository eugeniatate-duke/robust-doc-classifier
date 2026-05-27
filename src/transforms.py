from torchvision import transforms

# this file defines 2 pipelines of how the images are prepared before enterign the model


# pipeline 1: simple preprocessing. Resizes images, converts to tensor, normalize 
def get_baseline_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

# pipeline 2: simulates real-world messy uploads with bad lighting, blur, bad cropping etc 
def get_augmented_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])