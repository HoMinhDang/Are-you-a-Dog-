import torch

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss    

def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    num_epochs: int = 10
):
    print ("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}")


from model import SimpleCNNDog
from dataset import get_train_loader, get_test_loader        
if __name__ == "__main__":
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = SimpleCNNDog().to(device)
    train_loader = get_train_loader("data/train")
    test_loader = get_test_loader("data/test")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, test_loader, criterion, optimizer, device)
    
    save_path = "best_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")