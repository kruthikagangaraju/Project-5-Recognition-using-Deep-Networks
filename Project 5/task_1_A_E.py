# CS 5330
# Malhar Mahant & Kruthika Gangaraju
# Project 5: Recognition using Deep Networks
import sys
import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


# Function that performs the task 1 A
def task_1_a(train_loader):
    # Create a grid of plots to display the first six examples
    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(train_loader.dataset.classes[example_targets[i]]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# Class that defines the network to be used for MNIST, Greek Letters and FashionMNIST dataset
class MyNetwork(nn.Module):
    def __init__(self, learning_rate=0.01, kernel_size=5, dropout_rate=0.5):
        super(MyNetwork, self).__init__()
        self.learning_rate = learning_rate
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.momentum = 0.5
        self.log_interval = 10
        self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
        self.conv2_drop = nn.Dropout(p=dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    # computes a forward pass for the network
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.conv2_drop(self.pool2(self.relu2(self.conv2(x))))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        return x

    # Helper method for training the network (Task 1 D)
    def train_network(self, optimizer, epoch, train_loader, train_losses, train_counter, train_accs,
                      model_save='model.pth', verbose=True):
        self.train()
        train_correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = self(data)
            loss = F.nll_loss(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                if verbose:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
                train_acc = train_correct / len(train_loader.dataset)
                train_accs.append(train_acc)
                # Task 1 E
                torch.save(self.state_dict(), model_save)
                torch.save(optimizer.state_dict(), 'optimizer.pth')
        return train_losses, train_counter, train_accs

    # Helper method for testing the network with the given dataset
    def test_network(self, test_loader, test_losses, test_accs):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                output = self(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        test_acc = correct / len(test_loader.dataset)
        test_accs.append(test_acc)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * test_acc))

        return test_losses, test_accs


# Method to train model, test it using the given dataset and plot the loss and accuracy over number of examples seen
# for both training and testing data
def train_network(model, train_loader, test_loader, n_epochs=3, model_save='model.pth', plot=True):
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
    train_acc = []
    test_acc = []
    optimizer = optim.SGD(model.parameters(), lr=model.learning_rate)
    model.test_network(test_loader, test_losses, test_acc)
    for epoch in range(1, n_epochs + 1):
        train_losses, train_counter, train_acc = model.train_network(optimizer, epoch, train_loader, train_losses,
                                                                     train_counter, train_acc, model_save, plot)
        test_losses, test_acc = model.test_network(test_loader, test_losses, test_acc)

    if plot:
        print(len(train_losses))
        print(len(train_acc))
        print(len(test_losses))
        print(len(test_acc))
        print(train_counter)
        print(test_counter)
        # Plot the training and testing loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_counter, train_losses, label='Training Loss')
        plt.plot(test_counter, test_losses, label='Testing Loss', marker='o', ls='')
        plt.axis()
        plt.title('Training and Testing Loss.')
        plt.figtext(.8, .6,
                    f"L = {model.learning_rate}\nK = {model.kernel_size}\nD = {model.dropout_rate}\nB = {train_loader.batch_size}\nE = {n_epochs}")
        plt.xlabel('Number of training examples seen')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Plot the training and testing accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(train_counter, train_acc, label='Training Accuracy')
        plt.plot(test_counter, test_acc, label='Testing Accuracy', marker='o', ls='')
        plt.title('Training and Testing Accuracy')
        plt.figtext(.8, .6,
                    f"L = {model.learning_rate}\nK = {model.kernel_size}\nD = {model.dropout_rate}\nB = {train_loader.batch_size}\nE = {n_epochs}")
        plt.xlabel('Number of training examples seen')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
    return train_losses, train_counter, train_acc, test_losses, test_counter, test_acc


# main function
# Runs the tasks 1.A to 1.E
def main(argv):
    # Task 1 B
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    batch_size_train = 64
    batch_size_test = 1000
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    # Task 1 C
    network = MyNetwork()
    task_1_a(train_loader)
    print(network)
    # Task 1 D
    train_network(network, train_loader, test_loader, 5, 'model.pth')
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
