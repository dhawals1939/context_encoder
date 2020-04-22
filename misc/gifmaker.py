import imageio
import glob
from tqdm import tqdm
from image_splitter import image_spliter
from pygifsicle import optimize

#make suitable changes to directory locations and numericals according to your batchsizes
def gifmaker():
    with tqdm(total=256) as t:
        for i in range(1, 257):
            images = list()
            filenames = sorted(glob.glob('split_images/%d/*.jpg'%i))

            with imageio.get_writer('gifs/%d.gif'%(i), mode='I') as writer:
                for f in filenames:
                    writer.append_data(imageio.imread(f))

            optimize('gifs/%d.gif'%(i))
            t.update()

image_spliter()
gifmaker()
