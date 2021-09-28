import torch
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from torchvision.utils import save_image
from torchvision import transforms

#hyperparameters
batch_size=32
image_size = 28*28
hidden_neurons= 256
latent_size = 64
learning_rate = 1e-4
epochs = 100

#image pre-process
transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean = 0.5, std = 0.5)]) # to make a distribution between [-1, 1]

#dataset
dataset = MNIST(root='data',
    train=True,
    download = True,
    transform=transform,
    )


#dataloaders
loader = DataLoader(dataset, batch_size=batch_size, shuffle = True)

#definition of the networks
D = nn.Sequential(
    nn.Linear(image_size, hidden_neurons),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_neurons, hidden_neurons),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_neurons, 1),
    nn.Sigmoid()
    )
G = nn.Sequential(
    nn.Linear(latent_size, hidden_neurons),
    nn.ReLU(),
    nn.Linear(hidden_neurons, hidden_neurons),
    nn.ReLU(),
    nn.Linear(hidden_neurons, image_size),
    nn.Tanh())

#Assumes cuda is avaiable
D = D.cuda()
G = G.cuda()

#Loss
loss_fn = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr = learning_rate)
g_optimizer = torch.optim.Adam(G.parameters(), lr = learning_rate)

#denormalization
def denorm(x):
    ans = (x*2./255) + ds_mean
   
#values to save
d_losses = torch.zeros(epochs)
g_losses = torch.zeros(epochs)

#real and fake labels
real_labels = torch.ones(batch_size, 1).cuda()
fake_labels = torch.zeros(batch_size, 1).cuda()

#train loop
for epoch in range(epochs):
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()
    print ('new epoch')
    for batch, (real_imgs, _) in enumerate(loader):

        #DISCRIMINATOR
        
        #loss from real images
        real_imgs = real_imgs.view(batch_size, -1).cuda()
        real_d_outputs = D(real_imgs)
        real_loss = loss_fn (real_d_outputs, real_labels)
        
        #loss from fake images
        z = torch.randn(batch_size, latent_size).cuda()
        fake_imgs = G(z)
        fake_d_outputs = D(fake_imgs)
        fake_loss = loss_fn(fake_d_outputs, fake_labels)
        
        #total loss
        d_loss = real_loss + fake_loss
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        #GENERATOR
        
        #latents
        z = torch.randn(batch_size, latent_size).cuda()
        fake_imgs = G(z)
        d_outputs = D(fake_imgs)
        g_loss = loss_fn(d_outputs, real_labels)
        
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
    if epoch%2 == 0:
        fake_imgs = fake_imgs.view(-1, 1, 28, 28)
        real_imgs = real_imgs.view(-1, 1, 28, 28)
        print('g_loss was {}'.format(g_loss.data.item()))
        print('d_loss was {}'.format(d_loss.data.item()))
        save_image(fake_imgs, '001/'+ 'fake_images-{}.png'.format(epoch))
        #save_image(real_imgs, '001/'+ 'real_images-{}.png'.format(epoch))
    
'''
figure = plt.figure(figsize=(3,3))
for i in range(1, 11):
    idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[idx]
    figure.add_subplot(2, 5, i)
    plt.title(label)
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.savefig('image_visualization/1.png')
'''