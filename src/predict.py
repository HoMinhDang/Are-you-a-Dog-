# Function to load model and make predictions
import torch
from PIL import Image
import torch.nn.functional as F

def predict_one_image(
    model,
    image_path,
    transform,
    device
):
    model.eval()
    
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        
        if transform is not None:
            img = transform(img)
        
        img = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(img)
            probs = F.softmax(logits, dim=1)   # <-- QUAN TRỌNG
            
            conf, pred = torch.max(probs, dim=1)
    
    return pred.item(), conf.item()

def predict_batch_images(
    model,
    image_paths,
    transform,
    device
):
    model.eval()
    all_preds = []
    
    for image_path in image_paths:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if transform is not None:
                img = transform(img)
            img = img.unsqueeze(0)  # Add batch dimension
            img = img.to(device)
            
            with torch.no_grad():
                outputs = model(img)
                _, pred = torch.max(outputs, 1)
                all_preds.append(pred.item())
    
    return all_preds
