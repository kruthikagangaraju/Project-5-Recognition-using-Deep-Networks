# CS 5330
# Malhar Mahant & Kruthika Gangaraju
# Project 5: Recognition using Deep Networks
import sys
import torch
import torchvision
import matplotlib.pyplot as plt

from task_1_A_E import MyNetwork


# Defines the transformation to convert the image to greyscale and crop it to 28x28
class GreyScaleTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        # x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return x  # torchvision.transforms.functional.invert(x)


# main function
# Runs the Task 1 G
def main(argv):
    # Make network code repeatable
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    network = MyNetwork()
    batch_size_test = 10

    # Load Custom Handwritten test data
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('Resized Handwritten Numbers/', transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            GreyScaleTransform(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])),
        batch_size=batch_size_test)

    network.load_state_dict(torch.load('model.pth'))

    network.eval()

    predictions = []
    images, labels = next(iter(test_loader))
    with torch.no_grad():
        for i in range(10):
            image, label = images[i], labels[i]
            output = network(image)
            predictions.append(output)
            print('Example {}: True Label: {}, Predicted Label: {}'.format(
                i + 1, label, torch.argmax(output)))
    fig, axs = plt.subplots(4, 3, figsize=(10, 8))
    for i, ax in enumerate(axs.flat):
        if i < 10:
            ax.imshow(images[i].numpy().squeeze(), cmap='gray')
            ax.set_title(f"True: {labels[i]}, Predicted: {torch.argmax(predictions[i])}")
            ax.axis('off')
    for ax in axs.flat[10:]:
        ax.remove()
    plt.show()
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv)
