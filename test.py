import torch 
from data_loader import image_loader
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from generator import generator as Generator
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

mask_x=mask_y=64
img_x, img_y =128,128

transforms_ =   [
                    transforms.Resize((img_x, img_y)),
                    transforms.ToTensor(),
                    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
                ]

generator_path = sys.argv[1]

test_path = sys.argv[2]

output_path = sys.argv[3]

num_files = len(glob.glob(test_path + '*.png'))

val_data_loader = DataLoader(
                                image_loader(path=test_path + '*.png', transforms_=transforms_, mode='val'),
                                batch_size=num_files,
                                shuffle=False,
                                num_workers=10
                             )

generator = Generator()
generator.load_state_dict(torch.load(generator_path))
generator.cuda()
Tensor = torch.cuda.FloatTensor

def save_sample():
    global val_data_loader
    samples, masked_samples, i = next(iter(val_data_loader))
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    i = i[0].item()
    gen_mask = generator(masked_samples)
    filled_samples = masked_samples.clone()
    filled_samples[:, :, i:i+mask_x, i:i+mask_y] = gen_mask
    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
    for i in range(sample.shape[0]):
        save_image(sample[i,:,:,:], output_path + 'im-%d.jpg'%(i), nrow=6, normalize=True)

save_sample()
