# CS 5330
# Malhar Mahant & Kruthika Gangaraju
# Project 5: Recognition using Deep Networks
import sys

import numpy as np
import torch
import cv2
import torchvision
import matplotlib.pyplot as plt

from task_1_A_E import MyNetwork


# Helper method to visualize the filter weights and print weights to console. (Task 2 A)
def plot_filter_weights(layer):
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(10, 8))
    for i in range(len(layer.weight)):
        print(f"Filter {i} Weight Shape")
        print(layer.weight[i, 0].shape)
        print(f"Filter {i} Weights")
        print(layer.weight[i, 0])

        filter_weights = layer.weight[i, 0].data.numpy()

        row = i // 4
        col = i % 4
        if row < 3 and col < 4:
            axs[row, col].imshow(filter_weights)
            plt.subplot(3, 4, i + 1)
            plt.title(f"Filter: {i}")
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
    # remove unused axes
    for ax in axs.flat[len(layer.weight):]:
        ax.remove()
    plt.show()


# Helper method to plot the filters and the corresponding filtered image (Task 2 B)
def plot_filtered_images(img, layer):
    # plot the filtered images
    fig, axs = plt.subplots(5, 4, figsize=(10, 8))
    for i, ax in enumerate(axs.flat):
        if i % 2 == 0:
            ax.imshow(layer.weight[i // 2, 0].data.numpy(), cmap='gray')
            plt.subplot(5, 4, i + 1)
            plt.title(f"Filter: {i // 2}")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            filter_img = cv2.filter2D(img, -1, layer.weight[i // 2, 0].data.numpy())
            ax.imshow(np.squeeze(filter_img), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()


# main function
# Runs the Task 2
def main(argv):
    # Make network code repeatable
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    network = MyNetwork()
    # Load training data
    train_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.1307,), (0.3081,))
                                              ]))

    with torch.no_grad():
        # load the saved model from file
        network.load_state_dict(torch.load('model.pth'))
        print(network)
        print("Filters Weight Shape")
        print(network.conv1.weight.shape)
        plot_filter_weights(network.conv1)

        # Get first image
        img = np.array(train_dataset[0][0])

        plot_filtered_images(img, network.conv1)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv)
