import os
import sys
import glob
from tqdm import tqdm
from PIL import Image

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from torch.autograd import Variable
from generator import generator as Generator
from torchvision import transforms

from data_loader import image_loader

mask_x=mask_y=64
img_x, img_y = 128, 128

transforms_ = [
                transforms.Resize((img_x, img_y)),
                transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5))
              ]

check_point_path = sys.argv[1]

test_path = sys.argv[2]

output_path = sys.argv[3]

num_files = len(glob.glob(test_path + '*.png'))

test_data_loader = DataLoader(
                                image_loader(path=test_path + '*.png', transforms_=transforms_, mode='test'),
                                batch_size=num_files,
                                shuffle=False,
                                num_workers=10
                             )

generator = Generator()
generator.load_state_dict(torch.load(check_point_path)['generator_state_dict'])


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

if torch.cuda.is_available():
    generator.cuda()

def save_sample():
    global test_data_loader
    imgs, masked_imgs, crop_ind = next(iter(test_data_loader))
    
    imgs = Variable(imgs.type(Tensor))
    
    masked_imgs = Variable(masked_imgs.type(Tensor))
    
    crop_ind = crop_ind[0].item() #As center crops Hardcoding
    
    generated_patches = generator(masked_imgs)

    filled_imgs = masked_imgs.clone()
    filled_imgs[:, :, crop_ind:crop_ind+mask_x, crop_ind:crop_ind+mask_y] = generated_patches
    
    samples = torch.cat((masked_imgs.data, filled_imgs.data, imgs.data), -2)
    with tqdm(total=samples.shape[0]) as t:
        for i in range(samples.shape[0]):
            save_image(samples[i, :, :, :], output_path + 'generated_imgs-%d.jpg'%(i+1), nrow=6, normalize=True)
            t.update()

save_sample()
