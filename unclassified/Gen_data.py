#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/7/30 10:10
 =================知行合一=============
 渠道排名分析数据模拟
'''
import random
import time

# 渠道名称、新增用户、新增日期、是否购车、购车信息、客户标识、客户信息、客户行为信息…
# 渠道:微信，车展，官网，论坛
# 车展
def gen_data(num=10000,user_ids=set()):
    #渠道名称
    # source = random.choice(['微信','车展','官网','论坛'])
    newer= 'user_'+str(random.randint(0,num)).zfill(get_num(num))
    gender = random.choice(['男','女'])
    flag = 1 if newer in user_ids else 0
    user_ids.add(newer)
    newly_date = gen_date()
    is_buy=random.choice(['是','否','是','是','是'])
    buy_info=''
    if '是'==is_buy:
        buy_info = '购车信息_'+ str(random.randint(0,num)).zfill(get_num(num))
    symbol = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    user_info = '客户信息_'+ str(random.randint(0,num)).zfill(get_num(num))
    return '{},{},{},{},{},{},{},{}'.format('车展',newer,gender,
                                         newly_date,
                                         is_buy,
                                         buy_info,
                                         symbol,
                                         user_info),flag
# 论坛
def gen_data2(num=10000,user_ids=set()):
    #渠道名称
    # source = random.choice(['微信','车展','官网','论坛'])
    newer= 'user_'+str(random.randint(0,num)).zfill(get_num(num))
    gender = random.choice(['男','女'])
    flag = 1 if newer in user_ids else 0
    user_ids.add(newer)
    newly_date = gen_date()
    is_buy=random.choice(['是','否','否','是','是'])
    buy_info=''
    if '是'==is_buy:
        buy_info = '购车信息_'+ str(random.randint(0,num)).zfill(get_num(num))
    symbol = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    user_info = '客户信息_'+ str(random.randint(0,num)).zfill(get_num(num))
    return '{},{},{},{},{},{},{},{}'.format('论坛',newer,gender,
                                         newly_date,
                                         is_buy,
                                         buy_info,
                                         symbol,
                                         user_info),flag
# 官网
def gen_data3(num=10000,user_ids=set()):
    #渠道名称
    # source = random.choice(['微信','车展','官网','论坛'])
    newer= 'user_'+str(random.randint(0,num)).zfill(get_num(num))
    gender = random.choice(['男','女'])
    flag = 1 if newer in user_ids else 0
    user_ids.add(newer)
    newly_date = gen_date()
    is_buy=random.choice(['是','否','否','否','是'])
    buy_info=''
    if '是'==is_buy:
        buy_info = '购车信息_'+ str(random.randint(0,num)).zfill(get_num(num))
    symbol = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    user_info = '客户信息_'+ str(random.randint(0,num)).zfill(get_num(num))
    return '{},{},{},{},{},{},{},{}'.format('官网',newer,gender,
                                         newly_date,
                                         is_buy,
                                         buy_info,
                                         symbol,
                                         user_info),flag
# 微信
def gen_data4(num=10000,user_ids=set()):
    #渠道名称
    # source = random.choice(['微信','车展','官网','论坛'])
    newer= 'user_'+str(random.randint(0,num)).zfill(get_num(num))
    gender = random.choice(['男','女'])
    flag = 1 if newer in user_ids else 0
    user_ids.add(newer)
    newly_date = gen_date()
    is_buy=random.choice(['是','否','否','否','否'])
    buy_info=''
    if '是'==is_buy:
        buy_info = '购车信息_'+ str(random.randint(0,num)).zfill(get_num(num))
    symbol = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    user_info = '客户信息_'+ str(random.randint(0,num)).zfill(get_num(num))
    return '{},{},{},{},{},{},{},{}'.format('微信',newer,gender,
                                         newly_date,
                                         is_buy,
                                         buy_info,
                                         symbol,
                                         user_info),flag


def gen_date():
    a1 = (2017,1,1,0,0,0,0,0,0)
    a2 = (2018,5,31,23,59,59,0,0,0)
    start = time.mktime(a1) # 开始时间戳
    end = time.mktime(a2)
    num = random.randint(start,end)
    date = time.localtime(num)
    date = time.strftime('%Y-%m-%d %H:%M:%S', date)
    return date

def get_num(num):
    c = 0
    while num !=0:
        num = num//10
        c +=1
    return c


if __name__ == '__main__':
    count = 100000 #10w条数据
    user_set = set()
    with open('source_rank.txt','w',encoding='utf-8') as f:
        f.writelines('渠道名称,新增用户,性别,新增日期,是否购车,购车信息,客户标识,客户信息' + '\n')
        # for i in range(count):
        #     data,flag = gen_data(count,user_set)
        #     if(flag != 1):
        #         f.writelines(data+'\n')
        # 车展40000
        for i in range(40000):
            data, flag = gen_data(count, user_set)
            if (flag != 1):
                f.writelines(data+'\n')
        # 论坛30000
        for i in range(30000):
            data, flag = gen_data2(count, user_set)
            if (flag != 1):
                f.writelines(data + '\n')
        # 官网20000
        for i in range(20000):
            data, flag = gen_data3(count, user_set)
            if (flag != 1):
                f.writelines(data + '\n')
        # 微信10000
        for i in range(10000):
            data, flag = gen_data4(count, user_set)
            if (flag != 1):
                f.writelines(data + '\n')

    print(get_num(count))