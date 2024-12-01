import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# **1. Check for GPU Availability**
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# **2. Define the Simple CNN Model**
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # One convolutional layer
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # Pooling layer
        self.pool = nn.MaxPool2d(2)
        # One fully connected layer
        self.fc1 = nn.Linear(10 * 12 * 12, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Correct reshaping
        x = self.fc1(x)
        return x

# **3. Load the MNIST Dataset**
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

# **4. Initialize the Model, Loss Function, and Optimizer**
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# **5. Training Loop**
model.train()
for epoch in range(1):  # One epoch is sufficient for testing
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Move data to the appropriate device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        # Print loss every 100 batches
        if batch_idx % 100 == 99:
            print(f'Batch {batch_idx + 1}, Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print('Training completed successfully.')
