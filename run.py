import torch
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

training_data = MNIST(root='data',
    train=True,
    download = True,
    )

figure = plt.figure(figsize=(3,3))
for i in range(1, 11):
    idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[idx]
    figure.add_subplot(2, 5, i)
    plt.title(label)
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.savefig('image_visualization/1.png')