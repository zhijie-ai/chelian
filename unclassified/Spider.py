#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/8/21 13:31
 =================知行合一=============
'''

import re
import requests
import threading,datetime
from bs4 import BeautifulSoup
import random

# import requests
# requests.get('http://www.dict.baidu.com/s', params={'wd': 'python'})    #GET参数实例
# requests.post('http://www.itwhy.org/wp-comments-post.php', data={'comment': '测试POST'})    #POST参数实例


def write(path,text):
    with open(path,'a', encoding='utf-8') as f:
        f.writelines(text)
        f.write('\n')
# 清空文档
def truncatefile(path):
    with open(path, 'w', encoding='utf-8') as f:
        f.truncate()
# 读取文档
def read(path):
    with open(path, 'r', encoding='utf-8') as f:
        txt = []
        for s in f.readlines():
            txt.append(s.strip())
    return txt
# ----------------------------------------------------------------------------------------------------------------------
# 计算时间差,格式: 时分秒
def gettimediff(start,end):
    seconds = (end - start).seconds
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    diff = ("%02d:%02d:%02d" % (h, m, s))
    return diff

def gethead():
    user_agent_list = [ \
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1" \
        "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11", \
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6", \
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6", \
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1", \
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5", \
        "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5", \
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3", \
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3", \
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3", \
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3", \
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3", \
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3", \
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3", \
        "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3", \
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3", \
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24", \
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24"
    ]
    UserAgent = random.choice(user_agent_list)
    header = {'User-Agent':UserAgent}
    return header

#检查IP是否可用
def checkip(targetUrl,ip):
    header=gethead()
    proxies = {'http':'http://'+ip,'https':'https://'+ip}
    try:
        response_status = requests.get(url=targetUrl,proxies=proxies,headers= header,timeout=5).status_code
        if 200==response_status:
            return True
        else:
            return False
    except:
        print('IP:{}无法访问'.format(ip))
        return False

def get_proxy_id(type,pageNum,targetUrl,path):# ip类型,页码,目标url,存放ip的路径
    list = {'1': 'http://www.xicidaili.com/nt/',  # xicidaili国内普通代理
            '2': 'http://www.xicidaili.com/nn/',  # xicidaili国内高匿代理
            '3': 'http://www.xicidaili.com/wn/',  # xicidaili国内https代理
            '4': 'http://www.xicidaili.com/wt/'}  # xicidaili国外http代理
    url = list[str(type)]+str(pageNum)
    headers = gethead()
    html = requests.get(url=url,headers=headers,timeout=5).text
    soup = BeautifulSoup(html,'lxml')
    all = soup.find_all('tr',class_='odd')
    for i in all:
        t = i.find_all('td')
        ip2 = '"\'' + t[5].text.lower() + '\'' + ":'" + t[5].text.lower() + '://' + t[1].text + ':' + t[2].text + '\'",'
        ip = t[1].text+':'+t[2].text
        is_avail  = checkip(targetUrl,ip)
        if is_avail:
            write(path=path,text = ip2)
            print(ip)

def get_proxy_id2(type,targetUrl,path):# ip类型,页码,目标url,存放ip的路径
    list = {'1': 'http://www.xicidaili.com/nt/',  # xicidaili国内普通代理
            '2': 'http://www.xicidaili.com/nn/',  # xicidaili国内高匿代理
            '3': 'http://www.xicidaili.com/wn/',  # xicidaili国内https代理
            '4': 'http://www.xicidaili.com/wt/'}  # xicidaili国外http代理

    headers = gethead()

    for j in range(10):
        url = list[str(type)] + str(j+1)
        html = requests.get(url=url, headers=headers, timeout=5).text
        soup = BeautifulSoup(html, 'lxml')
        all = soup.find_all('tr', class_='odd')
        for i in all:
            t = i.find_all('td')
            ip2 = '"\'' + t[5].text.lower() + '\'' + ":'" + t[5].text.lower() + '://' + t[1].text + ':' + t[
                2].text + '\'",'
            ip = t[1].text + ':' + t[2].text
            is_avail = checkip(targetUrl, ip)
            if is_avail:
                write(path=path, text=ip2)
                print(ip)

def getip(targetUrl,path):
    truncatefile(path)
    start = datetime.datetime.now()
    threads =[]
    for type in range(4):# 四种类型ip,每种类型取前三页,共12条线程
        for pageNum in range(3):
            t = threading.Thread(target=get_proxy_id,args = (type+1,pageNum+1,targetUrl,path))
            threads.append(t)

    print('开始爬取代理ip')
    for s in threads:
        s.start()
    for e in threads:
        e.join()
    print('爬取完成....')
    end = datetime.datetime.now()
    diff = gettimediff(start,end)
    ips = read(path)
    print('一共爬取代理ip: %s 个,共耗时: %s \n' % (len(ips), diff))

def getip2(targetUrl,path):
    truncatefile(path)
    start = datetime.datetime.now()
    threads =[]
    for type in range(4):
        t = threading.Thread(target=get_proxy_id2,args = (type+1,targetUrl,path))
        threads.append(t)

    print('开始爬取代理ip')
    for s in threads:
        s.start()
    for e in threads:
        e.join()
    print('爬取完成....')
    end = datetime.datetime.now()
    diff = gettimediff(start,end)
    ips = read(path)
    print('一共爬取代理ip: %s 个,共耗时: %s \n' % (len(ips), diff))

if __name__ == '__main__':
    path='./data/ip.txt'
    targetUrl = 'http://www.cnblogs.com/TurboWay/'
    getip2(targetUrl,path)