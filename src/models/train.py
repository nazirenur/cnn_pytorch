import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.visualization import plot_sample_image
from models.model import CNNModel

def train_model(train_loader, test_loader, num_epochs, learning_rate):
    model = CNNModel()
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Convert the image data to float32
            train = Variable(images.view(-1, 1, 28, 28)).float()
            labels = Variable(labels)

            # Clear gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(train)

            # Calculate softmax and cross entropy loss
            loss = error(outputs, labels)

            # Calculate gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            count += 1

            if count % 50 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    # Convert the image data to float32
                    test = Variable(images.view(-1, 1, 28, 28)).float()

                    # Forward propagation
                    outputs = model(test)

                    # Get predictions from the maximum value
                    predicted = torch.max(outputs.data, 1)[1]

                    # Total number of labels
                    total += len(labels)
                    correct += (predicted == labels).sum()

                accuracy = 100 * correct / float(total)

                # Store loss and iteration
                loss_list.append(loss.item())
                iteration_list.append(count)
                accuracy_list.append(accuracy.item())

            if count % 500 == 0:
                # Print Loss
                print(f'Iteration: {count}  Loss: {loss.item()}  Accuracy: {accuracy.item()} %')

    return model, loss_list, iteration_list, accuracy_list