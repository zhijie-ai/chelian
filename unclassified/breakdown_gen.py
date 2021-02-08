#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/8/6 17:08
 =================知行合一=============
 故障分析数据模拟
'''
import random
import time

# 获取故障码及故障诱因
def get_break_code(dict,id):
    reason = dict.get(id)
    return reason

def get_count():
    with open('./data/故障码.csv') as f:
        dict = {line.strip().split(',')[0]:line.strip().split(',')[1] for line in f.readlines()}
        dict['0']='normal'
        return dict
# 产生车辆唯一VIN编码
def gen_VIN():
    data = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    result='LV'
    for i in range(15):
        result += random.choice(data)
    return result

# 获取发动机启动和停止时间,停止时间在启动时间5小时后的随机时间段内
def get_engine_start_end_time():
    a1 = (2017, 1, 1, 0, 0, 0, 0, 0, 0)
    a2 = (2017, 12, 31, 23, 59, 59, 0, 0, 0)
    start = time.mktime(a1)  # 开始时间戳
    end = time.mktime(a2)
    num = random.randint(start, end)
    engine_start = time.localtime(num)
    engine_start = time.strftime('%Y%m%d%H%M%S', engine_start)
    # 发动机启动5小时后
    end = num + 5*60*60
    end = random.randint(num, end)
    engine_stop = time.strftime('%Y%m%d%H%M%S',time.localtime(end))
    return engine_start,engine_stop

'''
    获取车辆的时间和行驶路程。行驶时间在0-5*60内，行驶速度在[0,120]之内
'''
def get_distance_and_time():
    interval = random.randint(0,300)
    distance = interval * random.uniform(0,3)
    return interval,distance

# 获取本次急加速次数，急减速次数，急转弯次数，本次怠速时长(车行驶中的停止时间，秒计)
def get_():
    d1 = random.randint(0,10)
    d2 = random.randint(0,10)
    d3 = random.randint(0,7)
    d4 = random.randint(0,5)
    return d1,d2,d3,d4


def tt():
    a1 = (2017, 1, 1, 0, 0, 0, 0, 0, 0)
    a2 = (2017, 12, 31, 23, 59, 59, 0, 0, 0)
    start = time.mktime(a1)  # 开始时间戳
    end = time.mktime(a2)
    num = random.randint(start, end)
    engine_start = time.localtime(num)
    engine_start = time.strftime('%Y%m%d%H%M%S', engine_start)
    # 发动机启动5小时后
    end = num + 5 * 60 * 60
    end = random.randint(num, end)
    engine_stop = time.strftime('%Y%m%d%H%M%S', time.localtime(end))
    return engine_start, engine_stop

def gen_data():
    #电池电压(100-220V)
    battery_voltage = random.randint(100,220)
    # 转向灯
    turn_ligth = random.choice(['开','关'])
    # 门锁
    clock = random.choice(['开','关'])
    # 天窗电机(kw)
    electrical_machine = random.randint(100,360)
    # 车窗(%)
    car_window = random.randint(0,100)
    # 左右前门
    left_forward_door = random.choice(['开','关'])
    rigth_forward_door = random.choice(['开','关'])
    # 变速箱温度
    gearbox_temperature = random.randint(100,150)
    # 动力系统转速(RPM,即转每分)
    rotate_speed = random.randint(4000,8500)
    # 巡航控制
    controller = random.choice(['开','关'])
    # 离合器温度(℃)
    clutch_temperature = random.randint(100,150)
    # ABS信号-紧急制动状态
    ABS_status = random.choice(['开','关'])
    # ABS信号-自动驻车信号：开/关
    ABS_stop = random.choice(['开','关'])
    #制动踏板信号：制动踏板百分比%
    brake = random.randint(0,100)
    # 档位信号：N、1、2、3、4、5、6、R，P、D、S或L档
    gears = random.choice(['N','1','2','3','4','5','6','R','P','D','S','L'])
    # 电源电压：V
    power_voltage = random.randint(100,220)
    # 离合器位置信号：%
    clutch_position = random.randint(0,100)
    # ECU电压：V
    ECU_voltage = random.randint(10,30)
    # 压力传感器信号：kpa
    pressure = random.randint(0,100)
    # 左前轮轮速传感器:km/h
    left_forward_transducer = random.randint(15,160)
    # 右前轮轮速传感器: km / h
    right_forward_transducer = random.randint(15,160)
    # 左后轮轮速传感器:km/h
    left_back_transducer = random.randint(15,160)
    # 右后轮轮速传感器:km/h
    right_back_transducer = random.randint(15,160)
    # 混风电机堵转：RPM
    mix_electrical_machine = random.randint(0,100)
    # 模式风门电机堵转：RPM
    pattern_electrical_machine = random.randint(0,100)
    return '{},{},{},{},{},{},{},{},{},{},{},{},{},' \
           '{},{},{},{},{},{},{},{},{},{},{},{}'.format(battery_voltage,turn_ligth,clock,
                                                     electrical_machine,car_window,
                                                     left_forward_door,rigth_forward_door,
                                                     gearbox_temperature,rotate_speed,
                                                     controller,clutch_temperature,
                                                     ABS_status,ABS_stop,brake,gears,
                                                     power_voltage,clutch_position,ECU_voltage,
                                                     pressure,left_forward_transducer,
                                                     right_forward_transducer,left_back_transducer,
                                                     right_back_transducer,mix_electrical_machine,
                                                     pattern_electrical_machine)



def generator():
    unique = set()
    dict = get_count()
    print(sorted(list(dict.keys())))
    print(dict['0'])
    with open('./data/breakdown_data.txt', 'w', encoding='utf-8') as f:
        header = 'VIN,发动机启动时间,发动机停止时间,本次行驶里程(km),本次行驶时长(min),本次急加速次数,' \
                 '本次急减速次数,本次急转弯次数,本次怠速时长(s),电池电压,转向灯,门锁,天窗电机(kw),' \
                 '车窗(%),左前门,右前门,变速箱温度(℃),动力系统转速(RPM),巡航控制,离合器温度(℃),' \
                 'ABS信号-紧急制动状态,ABS信号-自动驻车信号,制动踏板信号(%),档位信号,电源电压,离合器位置信号(%),' \
                 'ECU电压(V),压力传感器信号(kpa),左前轮轮速传感器(km/h),右前轮轮速传感器(km/h),' \
                 '左后轮轮速传感器(km/h),右后轮轮速传感器(km/h),混风电机堵转(RPM),模式风门电机堵转(RPM),故障码,故障诱因信号\n'
        f.writelines(header)
        for key in dict:
            count = random.randint(400,500)
            for i in range(1):
                VIN = gen_VIN()
                engine_start, engine_stop = get_engine_start_end_time()
                time, distance = get_distance_and_time()
                d1, d2, d3, d4 = get_()
                break_id, reason = key,dict[key]

                unique_val = VIN + break_id
                if unique_val not in unique:
                    data = gen_data()
                    line = '{},{},{},{:.2f},{},{},{},{},{},' \
                        .format(VIN, engine_start, engine_stop,
                                distance, time, d1, d2, d3, d4,data, break_id, reason)
                    print('---',line)
                    line = line+data+','+break_id+','+reason+'\n'
                    print(line)
                    # f.writelines(line)


if __name__ == '__main__':
    # print(tt())
    generator()
    # time, distance = get_distance_and_time()
    # print(time,'{:.2f}'.format(distance))
    dict = get_count()
    print(len(list(dict.keys())))

