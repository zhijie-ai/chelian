#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/5 18:18                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import json

with open('captions_train2014.json', "r", encoding="utf-8") as f:
    data = json.load(f)
    print(len(data))
    print(len(data['images']))
    print(len(data['info']))
    print(len(data['annotations']))
    print(len(data['licenses']))
    print(data.keys())
    # print(data['annotations'])
    print('AAAAAAAAAAAAAA')
    item = data['annotations'][3]
    print(type(item))
    print(type(data['annotations']))
    print('item',item)
    caption = item['caption'].strip().lower()
    caption = caption.replace('.', '').replace(',', '').replace("'", '').replace('"', '')
    caption = caption.replace('&', 'and').replace('(', '').replace(')', '').replace('-', ' ').split()
    caption = [w for w in caption if len(w) > 0]
    print(caption)
    print('BBBBBBBBBBBBBBBBB')
    print(len(data['annotations']))
    d=[it['image_id'] for it in data['annotations']]
    print(len(set(d)))
    print(len(data['annotations']))

