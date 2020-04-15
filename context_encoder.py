# -*- coding: utf-8 -*-

import os
import sys
import glob

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms

import torch
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from generator import generator as Generator
from discriminator import discriminator as Discriminator
from data_loader import image_loader

data_path = sys.argv[1] +'/'

results_path = sys.argv[2] + '/'

log_path = sys.argv[3] + '/'

check_points_path = sys.argv[4] + '/'

model_path = sys.argv[5] + '/'

test_path = sys.argv[6] + '/'

test_output_path = sys.argv[7] +'/'


img_x, img_y=128, 128
channels=3
mask_x, mask_y=64, 64
batch_size = 256
epochs=200

prev_epoch = 1

#required --> transforms  Resize, to-tensor-for-pytorch, normalize

transforms_ =   [
                    transforms.Resize((img_x, img_y)),
                    transforms.ToTensor(),
                    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
                ]


adversarial_loss = torch.nn.MSELoss()
pixelwise_loss = torch.nn.L1Loss()

generator = Generator()
discriminator = Discriminator()

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    pixelwise_loss.cuda()
    adversarial_loss.cuda()

train_data_loader = DataLoader(
                                image_loader(path= data_path + '*.JPG', transforms_=transforms_),
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=10
                              )

val_data_loader = DataLoader(
                                image_loader(path=data_path + '*.JPG', transforms_=transforms_, mode='val'),
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=10
                             )

optimizer_g = torch.optim.Adam(generator.parameters(), lr=.0001, betas=(.5, .999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=.0001, betas=(.5, .999))


try:
    check_point = torch.load(check_points_path + 'check_point.pt')
    if check_point is not None:
        prev_epoch = check_point['epoch'] + 1
        epochs = prev_epoch + 1501

        generator.load_state_dict(check_point['generator_state_dict'])
        discriminator.load_state_dict(check_point['discriminator_state_dict'])

        optimizer_g.load_state_dict(check_point['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(check_point['optimizer_d_state_dict'])

        adversarial_loss.load_state_dict(check_point['adversarial_loss_state_dict'])
        pixelwise_loss.load_state_dict(check_point['pixelwise_loss_state_dict'])

except:
    pass

import numpy as np
epoch_list = np.array(list(range(prev_epoch, epochs)))
check_points = epoch_list[np.where(epoch_list%50 == 1)]

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def save_sample(epoch):
    global val_data_loader
    samples, masked_samples, i = next(iter(val_data_loader))
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    i = i[0].item()
    gen_mask = generator(masked_samples)
    filled_samples = masked_samples.clone()
    filled_samples[:, :, i:i+mask_x, i:i+mask_y] = gen_mask
    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
    save_image(sample, results_path + 'epoch-%d.jpg'%(epoch), nrow=6, normalize=True)

try:
    if os.path.exists(log_path + 'log_loss_file.csv'):
        log_loss_file = open(log_path + 'log_loss_file.csv','a')
    else:
        raise Exception
except:
    log_loss_file = open(log_path + 'log_loss_file.csv', 'w')
    log_loss_file.write('Epoch, Generator Loss, Discriminator Loss\n')


import time 

for epoch in range(prev_epoch, 1 + epochs):
    epoch_start = time.time()
    epoch_generator_loss, epoch_discriminator_loss = .0, .0
    for i , (imgs, masked_imgs, masked_parts) in enumerate(train_data_loader):

        valid = Variable(Tensor(imgs.shape[0], *(1, 3, 3)).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], *(1, 3, 3)).fill_(.0), requires_grad=False)

        imgs = Variable(imgs.type(Tensor))
        masked_imgs = Variable(masked_imgs.type(Tensor))
        masked_parts = Variable(masked_parts.type(Tensor))

        #train generator
        optimizer_g.zero_grad()

        #generated parts
        generated_parts = generator(masked_imgs)

        discriminator_result = discriminator(generated_parts)

        g_adv = adversarial_loss(discriminator_result, valid)
        g_pixel = pixelwise_loss(generated_parts, masked_parts)

        #combined loss
        g_loss = .001*g_adv + .999*g_pixel

        g_loss.backward()

        optimizer_g.step()

        epoch_generator_loss +=  g_loss.item() * generated_parts.shape[0] 

        #train discriminator
        optimizer_d.zero_grad()

        real_loss = adversarial_loss(discriminator(masked_parts), valid)
        fake_loss = adversarial_loss(discriminator(generated_parts.detach()), fake)

        d_loss = .5 * (real_loss + fake_loss)

        d_loss.backward()

        optimizer_d.step()

        epoch_discriminator_loss += d_loss * generated_parts.shape[0]

        print(
                'Epoch %d/%d Batch %d/%d [D loss: %f] [G adv: %f, pixel: %f]'
                %(epoch, epochs, i+1, len(train_data_loader), d_loss.item(), g_adv.item(), g_pixel.item())
             )
        
        if i%len(train_data_loader) == 0:
            metrics = (epoch, epoch_generator_loss/len(train_data_loader), epoch_generator_loss/len(train_data_loader))
            log_metrics = '%d,' + '%f,'*(len(metrics)-1)
            log_loss_file.write(log_metrics[:-1]%metrics + '\n')
            log_loss_file.flush()

            batches_done = epoch * len(train_data_loader) + i

    if epoch in check_points:
        save_sample(epoch)
        check_point = {
                        'epoch':                        epoch,
                        'generator_state_dict' :        generator.state_dict(),
                        'discriminator_state_dict' :    discriminator.state_dict(),
                        'optimizer_g_state_dict' :      optimizer_g.state_dict(),
                        'optimizer_d_state_dict' :      optimizer_d.state_dict(),
                        'adversarial_loss_state_dict':  adversarial_loss.state_dict(),
                        'pixelwise_loss_state_dict':    pixelwise_loss.state_dict(),
                      }
        torch.save(check_point, check_points_path + 'check_point.pt')
    print('Epoch %d/%d -- time  %d'%(epoch, epochs, time.time()-epoch_start))

log_loss_file.close()

torch.save(generator.state_dict(), model_path + 'generator.pt')
torch.save(discriminator.state_dict(), model_path + 'discriminator.pt')
