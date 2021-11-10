#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/9/7 17:03                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import redis

def redis_scan():
    try:
        redisconn = redis.StrictRedis(host='172.16.11.202', port=6379, db=2)        #不同db需要修改
    except Exception as e:
        print("connect redis error")
        sys.exit(1)
    cursor = 1
    isNoe = True
    arr = []
    while cursor != 0:
        if isNoe:
            cursor = 0
            isNoe = False
        keys = redisconn.scan(cursor,match='recall_strategy_city*', count=2000)        #每次拿2000个key
        # time.sleep(0.05)
        if len(keys[1]) == 0:
            print("key scan finish")
        else:
            for key in keys[1]:
                key = bytes.decode(key)
                arr.append(redisconn.get(key).decode().split(','))
                cursor = keys[0]
    return arr[:10]

print(len(redis_scan()))