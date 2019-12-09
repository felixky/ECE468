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

epoch = 4 # Fill in
batch_size = 20 # Fill in
learning_rate = .0085 # Fill in

# Download Data

mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

# Set Data Loader(input pipeline)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=batch_size,shuffle=True)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

        self.conv1 = nn.Conv2d(1,4,3,padding=1)
        self.conv2 = nn.Conv2d(4,8,3,padding=1)
        self.conv3 = nn.Conv2d(8,16,3,padding=1)
        self.conv4 = nn.Conv2d(16,32,3,padding=1)
        self.conv5 = nn.Conv2d(32,64,3,padding=1)
        
        self.norm = nn.BatchNorm2d(4)
        self.norm2 = nn.BatchNorm2d(8)
        self.norm3 = nn.BatchNorm2d(16)
        self.norm4 = nn.BatchNorm2d(32)
        self.norm5 = nn.BatchNorm2d(64)        
        self.max = nn.MaxPool2d(2,2)


    def forward(self,x):
        # Put together your encoder network here
		# If you have multiple sequential layers, chain them together here
        out = F.tanh(self.conv1(x))
        out = F.tanh(self.norm(out))
        
        out = F.tanh(self.conv2(out))
        out = F.tanh(self.norm2(out))
        
        out = F.tanh(self.conv3(out))
        out = F.tanh(self.norm3(out))

        out = F.tanh(self.conv4(out))
        out = F.tanh(self.norm4(out))        

        out = F.tanh(self.conv5(out))
        out = F.tanh(self.norm5(out))          
        
        out = F.tanh(self.max(out))

        return out
    
encoder = Encoder().cuda()


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.deconv1 = nn.ConvTranspose2d(64,32,3,2,1,1)
        self.deconv2 = nn.ConvTranspose2d(32,16,3,2,1,1)
        self.deconv3 = nn.ConvTranspose2d(16,8,3,2,1,1)
        self.deconv4 = nn.ConvTranspose2d(8,4,3,2,1,1)
        self.deconv5 = nn.ConvTranspose2d(4,1,3,2,1,1)
        
        self.batch1 = nn.BatchNorm2d(32)
        self.batch2 = nn.BatchNorm2d(16)
        self.batch3 = nn.BatchNorm2d(8)
        self.batch4 = nn.BatchNorm2d(4)
        self.batch5 = nn.BatchNorm2d(1)
        
        self.max = nn.MaxPool2d(2,2)

    def forward(self,x):
        # Put together your encoder network here
		# If you have multiple sequential layers, chain them together here
        out = F.tanh(self.deconv1(x))
        out = self.batch1(out)
        
        out = F.tanh(self.deconv2(out))
        out = F.tanh(self.batch2(out))
        out = self.max(out)
        
        out = F.tanh(self.deconv3(out))
        out = F.tanh(self.batch3(out))
        out = self.max(out)
        
        out = F.tanh(self.deconv4(out))
        out = F.tanh(self.batch4(out))
        out = self.max(out)        

        out = F.tanh(self.deconv5(out))
        out = F.tanh(self.batch5(out))
        out = self.max(out) 

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

    torch.save([encoder,decoder],'./model/deno_autoencoder.pkl')
    print(loss)
    

for image,label in test_loader:
    image_n = torch.mul(image+0.25, 0.1 * noise)
    image = Variable(image).cuda()
    image_n = Variable(image_n).cuda()
    optimizer.zero_grad()
    output = encoder(image_n)
    output = decoder(output)
    loss = loss_func(output,image)
    loss.backward()
#        optimizer.step()

#    torch.save([encoder,decoder],'./model/deno_autoencoder.pkl')
print("test ",loss)    

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
