import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


def evaluate_model(
    model,
    dataloader,
    criterion,
    device
):    
    model.eval()
    running_loss = 0.0
    total_pred = 0
    correct_pred = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_pred += (preds == labels).sum().item()
            total_pred += labels.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_pred / total_pred    
    return epoch_loss, accuracy

def get_predictions(
    model,
    dataloader,
    device
):
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    
    return all_preds

def calc_metric(
    true_labels,
    pred_labels
):
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def print_classification_report(
    true_labels,
    pred_labels,
    class_names
):
    
    report = classification_report(
        true_labels, pred_labels,
        target_names=class_names,
        zero_division=0
    )
    print(report) 
    return report


from model import SimpleCNNDog
from dataset import get_test_loader
if __name__ == "__main__":
    model_path = "models\\checkpoints\\model_weights.pth"
    test_path = "data\\0 - dataset\\test"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNNDog().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loader = get_test_loader(test_path)
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_accuracy = evaluate_model(
        model,
        test_loader,
        criterion,
        device
    )
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    true_labels = []
    for _, labels in test_loader:
        true_labels.extend(labels.numpy())
    pred_labels = get_predictions(
        model,
        test_loader,
        device
    )
    class_names = ["notdogs", "dogs"]
    print_classification_report(
        true_labels,
        pred_labels,
        class_names
    )