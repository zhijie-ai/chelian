#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/9/7 14:18
 =================知行合一=============
'''
import random
import time

random.seed(3)

def get_num(num=10000):
    c = 0
    while num !=0:
        num = num//10
        c +=1
    return c


#生成用户信息，用户信息包括，用户ID，性别，出生年月，是否结婚，手机号
def random_user(num=100000,userIds=set()):
    user_id = 'user_' + str(random.randint(0, num)).zfill(get_num(num))
    if user_id in userIds:
        return
    userIds.add(user_id)
    gender = random.choice(['男', '女'])
    borth = random.choice([str(i).zfill(2) for i in range(1965,2005)])\
            +'-'+random.choice([str(i).zfill(2) for i in range(1,13)])\
            +'-'+random.choice([str(i).zfill(2) for i in range(1,31)])
    isMarry=random.choice(['是','否'])
    phone = random.choice(['153','136','171','150','189'])+''.join(random.choice('0123456789') for i in range(8))
    user_info = '客户信息_' + str(random.randint(0, num)).zfill(get_num(num))
    return '{},{},{},{},{},{}'.format(user_id,gender,borth,isMarry,phone,user_info)



def gen_date():
    a1 = (2017,1,1,0,0,0,0,0,0)
    a2 = (2018,5,31,23,59,59,0,0,0)
    start = time.mktime(a1) # 开始时间戳
    end = time.mktime(a2)
    num = random.randint(start,end)
    date = time.localtime(num)
    date = time.strftime('%Y-%m-%d %H:%M:%S', date)
    return date

# 标签：['重安全', '喜舒适', '享空间', '爱动力', '价格控','燃油省','通过强']
# 车展数据,转换率最高,type[0:30000],由于数据的特殊性，车展数据选的标签为:重安全,爱动力,喜舒适,通过强
def gen_exhibition_data(num=10000):
    newly_date = gen_date()
    is_buy=random.choice(['是','否','是','是','是'])
    buy_info=''
    if '是'==is_buy:
        buy_info = '购车信息_'+ str(random.randint(0,num)).zfill(get_num(num))
    symbol = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    number = random.randint(1,len(type))
    label = random.sample(type,number)
    label = '，'.join(label)
    method = random.choice(['文字','图片','音频','视频','文字','文字','图片'])
    return '{},{},{},{},{},{},{}'.format('车展',newly_date,
                                         is_buy,
                                         buy_info,
                                         symbol,label,method)
# 标签：['重安全', '喜舒适', '享空间', '爱动力', '价格控','燃油省','通过强']
# 论坛:喜舒适,享空间,价格控,燃油省
def gen_forum_data(num=10000):
    newly_date = gen_date()
    is_buy=random.choice(['是','否','否','是','是'])
    buy_info=''
    if '是'==is_buy:
        buy_info = '购车信息_'+ str(random.randint(0,num)).zfill(get_num(num))
    symbol = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    number = random.randint(1, len(type))
    label = random.sample(type,number)
    label = '，'.join(label)
    method = random.choice(['文字', '图片', '音频', '视频','文字'])
    return '{},{},{},{},{},{},{}'.format('论坛',newly_date,
                                         is_buy,
                                         buy_info,
                                         symbol,label,method)
# 官网
def gen_website_data(num=10000):
    newly_date = gen_date()
    is_buy=random.choice(['是','否','否','否','是'])
    buy_info=''
    if '是'==is_buy:
        buy_info = '购车信息_'+ str(random.randint(0,num)).zfill(get_num(num))
    symbol = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    number = random.randint(1, len(type))
    label = random.sample(type,number)
    label = '，'.join(label)
    method = random.choice(['文字', '图片', '音频', '视频','文字','视频'])
    return '{},{},{},{},{},{},{}'.format('官网',newly_date,
                                         is_buy,
                                         buy_info,
                                         symbol,label,method)

# 微信
def gen_weixin_data(num=10000):
    newly_date = gen_date()
    is_buy=random.choice(['是','否','否','否','是'])
    buy_info=''
    if '是'==is_buy:
        buy_info = '购车信息_'+ str(random.randint(0,num)).zfill(get_num(num))
    symbol = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    number = random.randint(1, len(type))
    label = random.sample(type,number)
    label = '，'.join(label)
    method = random.choice(['文字', '图片', '音频', '视频','图片','文字','视频'])
    return '{},{},{},{},{},{},{}'.format('微信',newly_date,
                                         is_buy,
                                         buy_info,
                                         symbol,label,method)

def to_file():
    # 不到13w条用户信息
    with open('./data/user_info.csv','w',encoding='utf-8') as f:
        header = '用户ID,性别,出生日期,是否结婚,手机号,用户信息'
        f.write(header+'\n')
        num = 200000
        for i in range(num):
            line = random_user(num,users)
            if line is None:
                continue
            f.write(line+'\n')

users = set()

type = ['重安全', '喜舒适', '享空间', '爱动力', '价格控','燃油省','通过强']
if __name__ == '__main__':
    to_file()
    users_list = list(users)
    print(len(users_list))
    with open('data/source.txt', 'w', encoding='utf-8') as f:
        f.writelines('用户ID,渠道,日期,是否购车,购车信息,客户标志,标签,投放方式' + '\n')
        for i in range(len(users_list[0:33000])):
            line = gen_exhibition_data()
            f.writelines(users_list[i]+','+line + '\n')
        for i in range(len(users_list[33000:65000])):
            line = gen_forum_data()
            f.writelines(users_list[i]+','+line + '\n')
        for i in range(len(users_list[65000:96000])):
            line = gen_website_data()
            f.writelines(users_list[i]+','+line + '\n')
        for i in range(len(users_list[96000:])):
            line = gen_weixin_data()
            f.writelines(users_list[i]+','+line + '\n')

