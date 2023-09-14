# CS 5330
# Malhar Mahant & Kruthika Gangaraju
# Project 5: Recognition using Deep Networks
import ast
import os
import sys
import itertools
import torch
import torchvision
import matplotlib.pyplot as plt

from task_1_A_E import train_network, MyNetwork


def plot_viewer(title):
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.show()


# Helper method to plot diffent graphs using the calculated result accuracies for combinations of hyperparameters
def plot_lines(epochs, results):
    learning_rate = [0.01, 0.05, 0.1, 0.5]
    dropout_rate = [0.1, 0.3, 0.5, 0.7]
    plt.figure(figsize=(10, 8))
    for hyperparams, result in results.items():
        k, d, l = hyperparams
        plt.plot(epochs, result, label=f'k={k}, d={d}, l={l}')
    plot_viewer('Accuracy over epochs for different hyperparameters')

    k = 5
    for d in dropout_rate:
        plt.figure(figsize=(10, 8))
        for l in learning_rate:
            plt.plot(epochs, results[(k, d, l)], label=f'k={k}, d={d}, l={l}')
        plot_viewer(f'Accuracy over epochs for different values of learning rate for dropout {d}')

    l = 0.1
    plt.figure(figsize=(10, 8))
    for d in dropout_rate:
        plt.plot(epochs, results[(k, d, l)], label=f'k={k}, d={d}, l={l}')
    plot_viewer(f'Accuracy over epochs for different values of dropout rate for learning rate {l}')

    # main function


# Runs the Task 4
def main(argv):
    # Make network code repeatable
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    batch_size_train = 64
    learning_rate = [0.01, 0.05, 0.1, 0.5]
    dropout_rate = [0.1, 0.3, 0.5, 0.7]
    kernel_size = [3, 5, 7, 9]
    epochs = [3, 5, 10, 15, 30]
    batch_size_test = 1000
    filename = 'results_4.txt'

    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            first_line = f.readline().strip()
            results = ast.literal_eval(first_line)
            print(results)
            print(results[(5, 0.5, 0.1)])
            plot_lines(epochs, results)
    else:
        print(f"The file '{filename}' does not exist.")
        # Load test data from FashionMNIST
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(root="data", train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor()
                                              ])),
            batch_size=batch_size_test, shuffle=True)

        # Load training data from FashionMNIST
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(root="data", train=True, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor()
                                              ])),
            batch_size=batch_size_train, shuffle=True)

        results = {}
        best_acc = 0
        best_params = None
        # Try all combinations
        for k, d, l in itertools.product(kernel_size, dropout_rate, learning_rate):
            try:
                network = MyNetwork(learning_rate=l, kernel_size=k, dropout_rate=d)
                model_name = f'm{k}_{d}_{l}.pth'
                print(network)
                print(model_name)
                *_, test_acc = train_network(network, train_loader, test_loader, epochs[-1], model_name, False)
                plot_test_acc = [test_acc[0]]
                for e in epochs:
                    plot_test_acc.append(test_acc[e - 1])
                results[(k, d, l)] = plot_test_acc
                if test_acc[-1] > best_acc:
                    best_acc = test_acc[-1]
                    best_params = (k, d, l)
            except:
                continue

        print(results)
        print(best_params)
        print(best_acc)
        with open(filename, 'a') as file:
            file.write(f'{results}\n')
            file.write(f'{best_params}\n')
            file.write(f'{best_acc}\n')
        plot_lines(epochs, results)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
