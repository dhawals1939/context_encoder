import cv2

import os
import glob
from tqdm import tqdm

#make suitable changes according to you batch size and directory locations

def image_spliter():
    files = glob.glob('./results/*jpg') #epoch results location

    for i in range(1,257):
        if not os.path.exists('split_images/%d'%(i)):
            os.mkdir('split_images/%d'%(i))

    with tqdm(total=len(files)) as t:
        for f in files:
            epoch = int(f.split('/')[-1].split('-')[-1].split('.jpg')[0])

            im = cv2.imread(f)

            tiles = [im[x:x+386, y:y+130, :] for x in range(0, im.shape[0], 386) for y in range(0, im.shape[1], 130) if y + 130 < im.shape[1]]
            [cv2.imwrite(('split_images/%d/%d.jpg')%(i+1, epoch), tiles[i]) for i in range(len(tiles))]
            t.update()
