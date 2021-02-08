#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/5 20:30                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
import json
import tqdm
import cv2
from imageio import imread

batch_size = 128
maxlen = 20
image_size = 224

MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3))


def load_data(image_dir, annotation_path):
    with open(annotation_path, 'r') as fr:
        annotation = json.load(fr)

    ids = []
    captions = []
    image_dict = {}
    for i in tqdm(range(len(annotation['annotations']))):
        item = annotation['annotations'][i]
        caption = item['caption'].strip().lower()
        caption = caption.replace('.', '').replace(',', '').replace("'", '').replace('"', '')
        caption = caption.replace('&', 'and').replace('(', '').replace(')', '').replace('-', ' ').split()
        caption = [w for w in caption if len(w) > 0]

        if len(caption) <= maxlen:
            if not item['image_id'] in image_dict:
                img = imread(image_dir + '%012d.jpg' % item['image_id'])
                h = img.shape[0]
                w = img.shape[1]
                if h > w:
                    img = img[h // 2 - w // 2: h // 2 + w // 2, :]
                else:
                    img = img[:, w // 2 - h // 2: w // 2 + h // 2]
                img = cv2.resize(img, (image_size, image_size))

                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                    img = np.concatenate([img, img, img], axis=-1)

                image_dict[item['image_id']] = img

            ids.append(item['image_id'])
            captions.append(caption)

    return ids, captions, image_dict


train_json = 'data/captions_train2014.json'
train_ids, train_captions, train_dict = load_data('data/train/images/COCO_train2014_', train_json)
print(len(train_ids))