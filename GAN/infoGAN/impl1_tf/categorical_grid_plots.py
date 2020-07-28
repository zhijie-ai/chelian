#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/3 17:00                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

# tf关于infoGAN的实现:https://github.com/JonathanRaiman/tensorflow-infogan


import numpy as np
import tensorflow as tf

from PIL import Image

def create_image_strip(images,zoom=1,gutter=5):
    num_images,image_height,image_width,channels = images.shape

    if channels == 1:
        images=images.reshape(num_images,image_height,image_width)

    # add gutter between images
    effective_collage_width = num_images*(image_width+gutter)-gutter

    # use white as backgroud
    start_color = (255,255,255)

    collage = Image.new('RGB',(effective_collage_width,image_height),start_color)
    offset = 0
    for image_idx in range(num_images):
        to_paste = Image.fromarray(
            (images[image_idx]*255).astype(np.uint8)
        )
        collage.paste(
            to_paste,
            box=(offset,0,offset+image_width,image_height)
        )
        offset += image_width+gutter

    if zoom != 1:
        collage = collage.resize(
            (
                int(collage.size(0)*zoom),
                int(collage.size[1]*zoom)
            ),
            Image.NEAREST
        )
    return np.array(collage)

class CategoricalPlotter(object):
    def __init__(self,
                 journalist,
                 categorical_cardinality,
                 num_continuous,
                 style_size,
                 generate,
                 row_size=10,
                 zoom = 2.0,
                 gutter=3):
        self._journalist = journalist
        self._gutter = gutter
        self.categorical_cardinality = categorical_cardinality
        self.style_size = style_size
        self.num_continuous = num_continuous
        self._generate = generate
        self._zoom = zoom

        self._placeholders={}
        self._image_summaries={}



if __name__ == '__main__':
    print(tf.__version__)

