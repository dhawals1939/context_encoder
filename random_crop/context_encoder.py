# -*- coding: utf-8 -*-

import os
import sys
from tqdm import tqdm


import torch
from torch import optim
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms

from generator import generator as Generator
from discriminator import discriminator as Discriminator
from data_loader import image_loader

data_path = sys.argv[1] +'/'

results_path = sys.argv[2] + '/'

log_path = sys.argv[3] + '/'

check_points_path = sys.argv[4] + '/'

model_path = sys.argv[5] + '/'

log_epoch = int(sys.argv[6])

img_x, img_y=128, 128
channels=3
mask_x, mask_y=64, 64
batch_size = 256
epochs=5002

prev_epoch = 1

#required --> transforms  Resize, to-tensor-for-pytorch, normalize

transforms_ = [
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
                                image_loader(path=data_path + '*.JPG', transforms_=transforms_),
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

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=.0001, betas=(.5, .999))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=.0001, betas=(.5, .999))


try:
    check_point = torch.load(check_points_path + 'check_point.pt')
    if check_point is not None:
        prev_epoch = check_point['epoch'] + 1
        epochs = prev_epoch + 3002

        generator.load_state_dict(check_point['generator_state_dict'])
        discriminator.load_state_dict(check_point['discriminator_state_dict'])

        generator_optimizer.load_state_dict(check_point['optimizer_g_state_dict'])
        discriminator_optimizer.load_state_dict(check_point['optimizer_d_state_dict'])

        adversarial_loss.load_state_dict(check_point['adversarial_loss_state_dict'])
        pixelwise_loss.load_state_dict(check_point['pixelwise_loss_state_dict'])

except:
    pass

import numpy as np
epoch_list = np.array(list(range(prev_epoch, epochs)))
check_points = epoch_list[np.where(epoch_list%log_epoch == 1)]

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def save_sample(epoch):
    global val_data_loader

    act_imgs, masked_imgs, croped_inds = next(iter(val_data_loader))                                    #loads a batch_size of objects

    act_imgs = Variable(act_imgs.type(Tensor))
    masked_imgs = Variable(masked_imgs.type(Tensor))
    
    generated_patches = generator(masked_imgs)

    croped_ind = croped_inds[0].item()                                                                  #same for all as we are dont center crop
    filled_imgs = masked_imgs.clone()
    filled_imgs[:, :, croped_ind:croped_ind+mask_x, croped_ind:croped_ind+mask_y] = generated_patches
    
    sample = torch.cat((masked_imgs.data, filled_imgs.data, act_imgs.data), -2)                         # masked, filled, actual images in a row
    
    save_image(sample, results_path + 'epoch-%d.jpg'%(epoch), nrow=6, normalize=True)

try:
    if os.path.exists(log_path + 'train_log_loss_file.csv') and os.path.exists(log_path + 'val_log_loss_file.csv'):
        train_log_loss_file = open(log_path + 'train_log_loss_file.csv', 'a')
        val_log_loss_file = open(log_path + 'val_log_loss_file.csv', 'a')
    else:
        raise Exception
except:
    train_log_loss_file = open(log_path + 'train_log_loss_file.csv', 'w')
    val_log_loss_file = open(log_path + 'val_log_loss_file.csv', 'w')

    train_log_loss_file.write('Epoch,Generator Loss,Discriminator Loss\n')
    val_log_loss_file.write('Epoch,Generator Loss\n')

pixel_loss_wt, adv_loss_wt = .999, .001

for epoch in range(prev_epoch, 1 + epochs):
    train_epoch_generator_loss, train_epoch_discriminator_loss = .0, .0
    imgs_generated = 0
    #Training Phase
    with tqdm(desc='Epoch {:<7d} Trained Batches'.format(epoch), total=len(train_data_loader)) as progress:
        for i, (imgs, masked_imgs, actual_patchs) in enumerate(train_data_loader):

            # For discriminator

            imgs = Variable(imgs.type(Tensor))
            masked_imgs = Variable(masked_imgs.type(Tensor))
            actual_patchs = Variable(actual_patchs.type(Tensor))

            # Train Generator

            generator_optimizer.zero_grad()

            # Generated Patches
            generated_patches = generator(masked_imgs)

            discriminator_result = discriminator(generated_patches)

            real_labels = Variable(Tensor(discriminator_result.shape).fill_(1.0), requires_grad=False)   # real-labels
            fake_labels = Variable(Tensor(discriminator_result.shape).fill_(.0), requires_grad=False)    # fake-labels

            adversarial_loss_g = adversarial_loss(discriminator_result, real_labels)                     # For training generator generated
                                                                                                         # patches should be considered real by discriminator
            pixelwise_loss_g = pixelwise_loss(generated_patches, actual_patchs)

            # Combined Loss
            generator_loss = adv_loss_wt * adversarial_loss_g + pixel_loss_wt *pixelwise_loss_g

            # Update Generator
            generator_loss.backward()
            generator_optimizer.step()


            # Train Discriminator

            discriminator_optimizer.zero_grad()

            real_loss = adversarial_loss(discriminator(actual_patchs), real_labels)                      # For training discriminator
                                                                                                         # actucal patches are considered as real and generated are as fake hence according loss is calculated
            fake_loss = adversarial_loss(discriminator(generated_patches.detach()), fake_labels)

            discriminator_loss = .5 * (real_loss + fake_loss)

            # Update Discriminator
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # For logging purpose
            current_batch_size = generated_patches.shape[0]
            train_epoch_generator_loss += generator_loss.item() * current_batch_size
            train_epoch_discriminator_loss += discriminator_loss.item() * current_batch_size
            imgs_generated += current_batch_size

            progress.set_postfix_str(
                                        s='Discriminator Loss %f Generator Loss %f'%(
                                            train_epoch_discriminator_loss / imgs_generated,
                                            train_epoch_generator_loss / imgs_generated),
                                        refresh=True
                                   )
            progress.update()

    # Log Epoch Metrics of Train Data
    train_metrics = (epoch, train_epoch_discriminator_loss/len(train_data_loader.dataset), train_epoch_generator_loss/len(train_data_loader.dataset))
    train_log_metrics = '%d,' + '%f,'*(len(train_metrics)-1)
    train_log_loss_file.write(train_log_metrics[:-1]%train_metrics + '\n')
    train_log_loss_file.flush()

    val_epoch_generator_loss = .0
    imgs_generated = 0
    #Validation Phase
    with tqdm(desc='Epoch %d Tested Batches'%(epoch), total=len(val_data_loader)) as progress:
        for i, (imgs, masked_imgs, crop_ind) in enumerate(val_data_loader): #val/testloader returns img, cropped imgs and left top corner coordinate of crop
            
            imgs = Variable(imgs.type(Tensor))
            masked_imgs = Variable(masked_imgs.type(Tensor))

            crop_ind = crop_ind[0].item()                                                               # For validation all are center cuts

            generated_patches = generator(masked_imgs)

            actual_patchs = imgs[:, :, crop_ind : crop_ind + mask_x, crop_ind : crop_ind + mask_y].clone()

            pixelwise_losss_g = pixelwise_loss(generated_patches, actual_patchs)

            discriminator_result = discriminator(generated_patches)

            real_labels = Variable(Tensor(discriminator_result.shape).fill_(1.0), requires_grad=False)   # real-labels

            adversarial_loss_g = adversarial_loss(discriminator_result, real_labels)

            generator_loss = pixel_loss_wt * pixelwise_loss_g + adv_loss_wt * adversarial_loss_g

            val_epoch_generator_loss += generator_loss.item() * generated_patches.shape[0]

            imgs_generated += generated_patches.shape[0]
            
            progress.set_postfix_str(
                                        s='Generator Loss %f'%(val_epoch_generator_loss / imgs_generated),
                                        refresh=True
                                   )
            progress.update()

    # Log Epoch Metrics of Validation Data
    val_metrics = (epoch, val_epoch_generator_loss / len(val_data_loader.dataset))
    val_log_metrics = '%d,%f'
    val_log_loss_file.write(val_log_metrics%val_metrics + '\n')
    val_log_loss_file.flush()

    # Save Model and first batch of Validation samples
    if epoch in check_points:
        save_sample(epoch)
        check_point = {
                            'epoch':                        epoch,

                            'generator_state_dict' :        generator.state_dict(),
                            'discriminator_state_dict' :    discriminator.state_dict(),
                            
                            'optimizer_g_state_dict' :      generator_optimizer.state_dict(),
                            'optimizer_d_state_dict' :      discriminator_optimizer.state_dict(),
                            
                            'adversarial_loss_state_dict':  adversarial_loss.state_dict(),
                            'pixelwise_loss_state_dict':    pixelwise_loss.state_dict(),
                      }

        torch.save(check_point, check_points_path + 'check_point.pt')

train_log_loss_file.close()
val_log_loss_file.close()

torch.save(generator.state_dict(), model_path + 'generator.pt')
torch.save(discriminator.state_dict(), model_path + 'discriminator.pt')
