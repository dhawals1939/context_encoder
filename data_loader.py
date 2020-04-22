import glob
import random
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class image_loader(Dataset):
    def __init__(self, transforms_=None, **args):
        
        # print(args)
        self.img_x = 128 if 'img_x' not in args.keys() else args['img_x']
        self.img_y = 128 if 'img_y' not in args.keys() else args['img_y']
        
        self.mask_x = 64 if 'mask_x' not in args.keys() else args['mask_x']
        self.mask_y = 64 if 'mask_y' not in args.keys() else args['mask_y']

        self.mode = 'train' if 'mode' not in args.keys() else args['mode']

        self.i = 0
        self.files = sorted(glob.glob(args['path']))

        # Uncomment if using validation set
        # val_split = int(len(self.files) *  .2) # 20% Data for Validation
        
        # if self.mode == 'train':
        #     self.files = self.files[:-val_split]
        # elif self.mode == 'val':
        #     self.files = self.files[-val_split:]
        # else:
        #     self.files = self.files[:]

        self.transform = transforms.Compose(transforms_) if transforms_ else None
    
    def random_crop(self, img):
        # x1, y1 = random.randint(0, self.img_x-self.mask_x), random.randint(0, self.img_y-self.mask_y) # uncomment if training random crops
        x1 = y1 = self.mask_x // 2                                              # for simple center cuts
        x2, y2 = x1 + self.mask_x, y1 + self.mask_y
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def center_crop(self, img):
        i = (self.img_x - self.mask_x)//2
        masked_img = img.clone()
        masked_img[:, i:i+self.mask_x, i:i+self.mask_y] = 1
        
        return masked_img, i
    
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert('RGB')
        img = self.transform(img)

        if self.mode == 'train':
            masked_img, aux = self.random_crop(img)
        else:
            masked_img, aux = self.center_crop(img)
        self.i += 1
        return img, masked_img, aux
        
    def __len__(self):
        return len(self.files)
