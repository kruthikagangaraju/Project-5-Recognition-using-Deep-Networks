# CS 5330
# Malhar Mahant & Kruthika Gangaraju
# Project 5: Recognition using Deep Networks
import sys
import torch
import torchvision
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from task_1_A_E import MyNetwork, train_network


# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# Defines the transformation to convert the image to greyscale and crop it to 28x28
class GreyScaleTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        # x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return x  # torchvision.transforms.functional.invert(x)


# Helper function to test the given network with the given dataset and calculate accuracy over the data
# Additionally prints each true and predicted label and plots first 9 examples from the test data
def test_dataset(greek_train, network, label_classes):
    # set the model to evaluation mode
    network.eval()
    predictions = []
    images = []
    labels = []

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(greek_train):
            images.extend(image)
            labels.extend(label)
            output = network(image)
            predictions.extend(output)

    correct = 0
    for i in range(len(predictions)):
        pred = predictions[i].data.max(0, keepdim=True)[1]
        # if hasattr(greek_train.dataset, 'classes'):
        print('Example {}: True Label: {}, Predicted Label: {}'.format(
                i + 1, label_classes[labels[i]], label_classes[pred]))
        # else:
        #     print('Example {}: True Label: {}, Predicted Label: {}'.format(
        #         i + 1, labels[i], pred))
        correct += pred.eq(labels[i].data.view_as(pred)).sum()
    test_acc = correct / len(greek_train.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(greek_train.dataset),
        100. * test_acc))

    fig, axs = plt.subplots(3, 3, figsize=(10, 8))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i].numpy().squeeze().squeeze(), cmap='gray')
        # if hasattr(greek_train.dataset, 'classes'):
        ax.set_title(
                f"True: {label_classes[labels[i]]}, Predicted: {label_classes[torch.argmax(predictions[i])]}")
        # else:
        #     ax.set_title(
        #         f"True: {labels[i]}, Predicted: {torch.argmax(predictions[i])}")
        ax.axis('off')
    plt.show()
    return


# main function
# Runs the Task 3 and Extension 2
def main(argv):
    # Task 3
    training_set_path = 'greek_train/'
    handwritten_test_set_path = 'Resized Handwritten Greek Letters/'
    handwritten_training_set_path = 'Additional Resized Handwritten Greek Letters/'

    # Make network code repeatable
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    network = MyNetwork()
    # load the saved model from file
    network.load_state_dict(torch.load('model.pth'))

    # DataLoader for the Greek data set
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(training_set_path,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))])),
        batch_size=5,
        shuffle=True)

    # DataLoader for the Handwritten Greek data set
    handwritten_test = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(handwritten_test_set_path,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreyScaleTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))])),
        batch_size=5,
        shuffle=True)

    # DataLoader for the Handwritten Additional characters Greek data set
    handwritten_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(handwritten_training_set_path,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreyScaleTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))])),
        batch_size=5,
        shuffle=True)

    # Freeze the network weights
    for param in network.parameters():
        param.requires_grad = False

    # Replace the last layer with a new Linear layer with three nodes
    network.fc2 = nn.Linear(50, 3)
    # Train the last layer
    network.fc2.requires_grad = True
    print(network)
    train_network(network, greek_train, greek_train, 80, 'greek_model.pth')

    # Testing given dataset
    test_dataset(greek_train, network, greek_train.dataset.classes)

    # Testing handwritten dataset
    test_dataset(handwritten_test, network, handwritten_test.dataset.classes)

    # Extension 2
    combined_dataset = ConcatDataset([handwritten_train.dataset, greek_train.dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=5, shuffle=True)

    network.fc2 = nn.Linear(50, 7)
    network.fc2.requires_grad = True
    print(network)
    train_network(network, combined_loader, combined_loader, 80, 'greek_model_additional_char.pth')
    test_dataset(greek_train, network, combined_loader.dataset.datasets[1].classes)
    test_dataset(handwritten_train, network, combined_loader.dataset.datasets[0].classes)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv)
