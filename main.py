import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

# Check if CUDA is available and set PyTorch to use GPU or CPU accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# Load the dataset
train_dataset = ImageFolder('path_to_train_dataset', transform=processor)
test_dataset = ImageFolder('path_to_test_dataset', transform=processor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the pretrained model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

# Modify the last layer
num_classes = len(train_dataset.classes)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

# Move the model to GPU
model = model.to(device)

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()

# Evaluate the model
model.eval()
running_corrects = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs.logits, 1)
        running_corrects += torch.sum(preds == labels.data)

# Save the model
torch.save(model.state_dict(), 'model.pth')