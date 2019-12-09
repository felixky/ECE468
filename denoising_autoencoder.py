import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Set Hyperparameters

epoch = 5 # Fill in
batch_size = 25 # Fill in
learning_rate = .0085 # Fill in

# Download Data

mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

# Set Data Loader(input pipeline)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

        self.conv1 = nn.Conv2d(1,64,3,padding=1)
        self.norm = nn.BatchNorm2d(64)
        self.max = nn.MaxPool2d(2,2)


    def forward(self,x):
        # Put together your encoder network here
		# If you have multiple sequential layers, chain them together here
        out = F.tanh(self.conv1(x))
        out = F.tanh(self.norm(out))
        out = F.tanh(self.max(out))

        return out
    
encoder = Encoder().cuda()


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()

        self.deconv1 = nn.ConvTranspose2d(64,1,3,2,1,1)
        self.batch = nn.BatchNorm2d(1)

    def forward(self,x):
        # Put together your encoder network here
		# If you have multiple sequential layers, chain them together here
        out = F.tanh(self.deconv1(x))
        out = F.tanh(self.batch(out))
 
        return out

decoder = Decoder().cuda()

# Noise

noise = torch.rand(batch_size,1,28,28)

# loss func and optimizer
# we compute reconstruction after decoder so use Mean Squared Error
# In order to use multi parameters with one optimizer,
# concat parameters after changing into list

parameters = list(encoder.parameters())+ list(decoder.parameters())
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(parameters, lr=learning_rate)

# load saved weights if they exist

try:
	encoder, decoder = torch.load('./model/deno_autoencoder.pkl')
	print("\n--------model restored--------\n")
except:
	pass

# train the encoder and decoder
for i in range(epoch):
    for image,label in train_loader:
        image_n = torch.mul(image+0.25, 0.1 * noise)
        image = Variable(image).cuda()
        image_n = Variable(image_n).cuda()
        optimizer.zero_grad()
        output = encoder(image_n)
        output = decoder(output)
        loss = loss_func(output,image)
        loss.backward()
        optimizer.step()

    #torch.save([encoder,decoder],'./model/deno_autoencoder.pkl')
    print(loss)

# check single image with noise and denoised

img = image[0].cpu()
input_img = image_n[0].cpu()
output_img = output[0].cpu()

origin = img.data.numpy()
inp = input_img.data.numpy()
out = output_img.data.numpy()

plt.imshow(origin[0],cmap='gray')
plt.show()

plt.imshow(inp[0],cmap='gray')
plt.show()

plt.imshow(out[0],cmap="gray")
plt.show()

print(label[0])
