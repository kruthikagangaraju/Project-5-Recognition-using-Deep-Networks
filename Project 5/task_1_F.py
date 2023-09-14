# CS 5330
# Malhar Mahant & Kruthika Gangaraju
# Project 5: Recognition using Deep Networks
import sys

import torch
import torchvision
import matplotlib.pyplot as plt

from task_1_A_E import MyNetwork


# main function
# Runs the Task 1 F
def main(argv):
    # Make network code repeatable
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    network = MyNetwork()
    batch_size_test = 10

    # Load test data
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)

    # load the saved model from file
    network.load_state_dict(torch.load('model.pth'))

    # set the model to evaluation mode
    network.eval()

    # run the model on the first 10 examples in the test set
    predictions = []
    images, labels = next(iter(test_loader))
    print('Calulated logits')
    with torch.no_grad():
        for i in range(10):
            image, label = images[i], labels[i]
            output = network(image.unsqueeze(0))
            predictions.append(output)
            print(output.numpy().squeeze().round(2))
    for i in range(10):
        print('Example {}: True Label: {}, Predicted Label Index: {}, Predicted Label: {}'.format(
            i + 1, labels[i], torch.argmax(predictions[i]), test_loader.dataset.classes[torch.argmax(predictions[i])]))
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i].numpy().squeeze(), cmap='gray')
        ax.set_title(f"True: {labels[i]}, Predicted: {torch.argmax(predictions[i])}")
        ax.axis('off')
    plt.show()
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv)
