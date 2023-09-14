# CS 5330
# Malhar Mahant & Kruthika Gangaraju
# Project 5: Recognition using Deep Networks
import sys

import numpy as np
import torch
import torchvision

from task_2 import plot_filter_weights, plot_filtered_images


# main function
# Runs the Extension 1
def main(argv):
    # Load AlexNet model
    model = torchvision.models.alexnet(pretrained=True)
    print(model)
    model.eval()

    # Load training data
    train_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       (0.1307,), (0.3081,))
                                               ]))

    # Get first image
    img = np.array(train_dataset[0][0])
    with torch.no_grad():
        # Get the first convolution layer of the model
        layer1 = model.features[0]
        print("Filters Layer 1 Weight Shape")
        print(layer1.weight.shape)
        plot_filter_weights(layer1)
        plot_filtered_images(img, layer1)

        # Get the second convolution layer of the model
        layer2 = model.features[3]
        print("Filters Layer 2 Weight Shape")
        print(layer2.weight.shape)
        plot_filter_weights(layer2)
        plot_filtered_images(img, layer2)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv)
