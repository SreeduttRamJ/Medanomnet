import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import uuid
from torchvision import transforms
from utils.model import MidLevelResNet50_LightCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_labels = ['covid19', 'glioma_tumor', 'liver_health', 'liver_tumor', 'lung_opacity',
                'meningioma_tumor', 'no_tumor', 'normal', 'pituitary_tumor', 'viral_pneumonia']

model = MidLevelResNet50_LightCNN(num_classes=len(class_labels))
model.load_state_dict(torch.load('models/best_model.pt', map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    tensor = tensor.cpu().numpy() * std + mean
    return np.clip(tensor, 0, 1).transpose(1, 2, 0)

def load_and_prepare_slit_mask(path):
    overlay = Image.open(path).convert("RGBA")
    datas = overlay.getdata()
    new_data = []
    for item in datas:
        # Make white pixels transparent; adjust if your slit masks use black or white as background
        if item[:3] == (255, 255, 255):  
            new_data.append((255, 255, 255, 0))  # transparent
        else:
            new_data.append(item)
    overlay.putdata(new_data)
    return overlay

# Load slit masks once
brain_slit = load_and_prepare_slit_mask("static/slits/brain.png")
liver_slit = load_and_prepare_slit_mask("static/slits/liver.png")
chest_slit = load_and_prepare_slit_mask("static/slits/chest.png")

# Map classes to slits
slit_map = {
    'glioma_tumor': brain_slit,
    'meningioma_tumor': brain_slit,
    'pituitary_tumor': brain_slit,
    'liver_tumor': liver_slit,
    'liver_health': liver_slit,
    'lung_opacity': chest_slit,
    'covid19': chest_slit,
    'viral_pneumonia': chest_slit
}

def apply_slit_to_image(image_pil, slit_mask):
    slit_resized = slit_mask.resize(image_pil.size)
    image_rgba = image_pil.convert("RGBA")
    combined = Image.alpha_composite(image_rgba, slit_resized)
    return combined.convert("RGB")

def process_image(img_path):
    image = Image.open(img_path).convert('RGB')

    # First predict on original image
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(prob, dim=1)
        pred_class = class_labels[pred_idx.item()]

    # If no tumor or healthy class, skip heatmap generation
    if pred_class in ['liver_health', 'no_tumor', 'normal']:
        return {
            "filename": f"/static/uploads/{os.path.basename(img_path)}",
            "heatmap_path": None,
            "anomaly_detected": False,
            "confidence": round(conf.item() * 100, 2),
            "classification": pred_class,
            "anomaly_location": None
        }

    # Apply slit overlay if available for this class
    slit_mask = slit_map.get(pred_class)
    if slit_mask:
        masked_img = apply_slit_to_image(image, slit_mask)
    else:
        masked_img = image  # fallback

    # Forward pass on masked image for feature extraction & heatmap
    masked_tensor = transform(masked_img).unsqueeze(0).to(device)
    with torch.no_grad():
        x = model.stem(masked_tensor)
        x1 = model.layer1(x)
        x2 = model.layer2(x1)
        x3 = model.layer3(x2)
        x3_up = torch.nn.functional.interpolate(x3, size=(28, 28), mode='bilinear', align_corners=False)
        features = torch.cat([x2, x3_up], dim=1)  # (1, 1536, 28, 28)

        output = model.classifier(features)
        prob = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(prob, dim=1)
        pred_class = class_labels[pred_idx.item()]

        # Generate anomaly heatmap
        heatmap = features.mean(dim=1, keepdim=True)
        heatmap = torch.nn.functional.interpolate(heatmap, size=(224, 224), mode='bilinear')
        heatmap_np = heatmap.squeeze().cpu().numpy()
        heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min())

    # Save heatmap overlay on original image (not masked image)
    input_np = denormalize(transform(image).cpu())
    heatmap_id = str(uuid.uuid4())[:8]
    heatmap_path = f"static/uploads/heatmap_{heatmap_id}.png"

    plt.figure(figsize=(5, 5))
    plt.imshow(input_np, alpha=0.8)
    plt.imshow(heatmap_np, cmap='jet', alpha=0.3)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return {
        "filename": f"/static/uploads/{os.path.basename(img_path)}",
        "heatmap_path": f"/{heatmap_path}",
        "anomaly_detected": True,
        "confidence": round(conf.item() * 100, 2),
        "classification": pred_class,
        "anomaly_location": "Region highlighted in heatmap"
    }
