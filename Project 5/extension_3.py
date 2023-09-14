# CS 5330
# Malhar Mahant & Kruthika Gangaraju
# Project 5: Recognition using Deep Networks
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


# Defines the model for CIFAR10 dataset classification
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.learning_rate = 0.1
        self.kernel_size = 5
        self.dropout_rate = 0.5
        self.momentum = 0.5
        self.log_interval = 10
        self.conv1 = nn.Conv2d(3, 10, kernel_size=self.kernel_size)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=self.kernel_size)
        self.conv2_drop = nn.Dropout(p=self.dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(500, 50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.conv2_drop(self.pool2(self.relu2(self.conv2(x))))
        num_channels = x.size(1)
        feature_map_size = x.size(2) * x.size(3)
        x = x.view(-1, num_channels * feature_map_size)
        x = self.relu3(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        return x


def train_network(model, optimizer, loss_function, epochs, train_loader, model_save_name):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                torch.save(model.state_dict(), model_save_name)
                torch.save(optimizer.state_dict(), 'optimizer.pth')
    print('Finished Training')


# Helper function to test the model
def test_network(model, loss_function, test_loader):
    accuracy = 0
    total = 0
    test_loss = 0
    predictions = []
    test_images = []
    test_labels = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            if not predictions:
                test_images.extend(images)
                test_labels.extend(labels)
                predictions.extend(outputs)
            test_loss += loss_function(outputs, labels).item()
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        test_loss /= len(test_loader)
        accuracy /= len(test_loader)
        print(f"Test Loss: {test_loss:.3f}.. "
              f"Test Accuracy: {accuracy:.3f}")
    return test_images, test_labels, predictions, accuracy, test_loss


def plot_test_images(test_images, test_labels, predictions, label_classes):
    fig, axs = plt.subplots(3, 10, figsize=(8, 8))
    for i, ax in enumerate(axs.flat):
        ax.imshow(test_images[i].numpy().squeeze().transpose((1, 2, 0)))
        ax.set_title(
            f"{label_classes[test_labels[i]]}: ({label_classes[torch.argmax(predictions[i])]})")
        ax.axis('off')
    plt.show()


def main(argv):
    model_file = 'cifar10_model.pth'

    # Define the transforms for the dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the dataset
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                               shuffle=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128,
                                              shuffle=False)

    # Initialize model
    net = Net()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epochs = 300
    # Load file if present else train model
    if os.path.isfile(model_file):
        net.load_state_dict(torch.load(model_file))
    else:
        train_network(net, optimizer, criterion, epochs, train_loader, model_file)

    test_images, test_labels, predictions, accuracy, test_loss = test_network(net, criterion, test_loader)
    plot_test_images(test_images, test_labels, predictions, test_loader.dataset.classes)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
