#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/9/16 14:45
 =================知行合一=============
'''

import requests
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

header={'User-Agent':
            'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24'}

form_data={
'buid'	:'youku',
'callback'	:'http://pay.youku.com/buy/memberconfirm.html?memberid=100002',
'formtoken'	:'2153C6024E74E1F3C280A9016AA3F907',
'jsToken'	:'HV01PAAZ0b0b3ba53b6d93af5b9df8460425fb4d',
'loginType':	'passport_pwd',
'mode'	:'embedded',
'needRecommend':	'true',
'passport':	'1222',
'password':	'698d51a19d8a12221ce581499d7b701668',
'pid'	:'0160317PLF000211',
'rememberMe'	:'true',
'sendCodeType':	'mobileCode',
'state':'false',
'UA':'110#WrKkAUkfkOhFsylBoUc8Uiy8kuFIQMkNMRXKmeUyckTW46m2NTKk9wrAKkT7MkkUMRykkMyemkkGMdskHTSUcPeWkkDIDKk6FRwkkPJwkTk84KkkWLBksPXhbUgeFUkwMRykkbmXkTTWPSUkNbvkkMy8kk2IAU2k4RBknTZeNRY44Kr2NTSkFVX8kssIbk2swRyTk1JXkFRW1KT8NsckkhWkkkutDwkkqPyckTmP/xsYgKcxhed/jWpXTuBR81GO2cuQAlWy8TmakKfY6BzfBnRa2KmA1MJHFxIkhQ7/82aTrcyIbRjIuG972fFDbcd2fGQPfUY2pO+OEzc1JLvwsTvsMfpDu3ciskT47Tmwj3kq44CcsQvJIT1CBvspX39BD3SkGLKIjTcw4wk3sL/RsRFkdyEODeBtWcMEu+c4RQvbKVYNxIGiCqHnkcaOs3aEs8PhKfwoBNcqKK23IkPD5uGb2QGjs6A+MOEFZWJtKjAgkY9o765ktCEUW22A…M6AvTv0xK9gj2m8LJBmiR/6PyJt0X/dT1GGl596gseQy0MvtmhfazeL7OLtZzZkCY95NSvq8gb7DglYVk2r1hC6kIdB2CsRYlr07MIivPN2ykDcd3qLdQMxWzduDsU1BgsZq7rYwONC5pWes/fHORLn1WgDqv6DgRztg48LNNQUCa58//iGLOkarGl0LqmBkXSwTvYBa0yjLYZTuLznVu9bTu8qVHhBhP9xr+FbuoL2YYATTZMzeMVEdyl89o7BkZ14tZcYzb3OnA2o8CDB4eM7aN+nisdgDRMDRSM3+Oteewc8eeeVSULqj6YrVL9Ow0gu99SIl4Jn8ch0i3Riq5pj6ZqdIp3qrWva1SLJDLgUgFN73+/X7WuF0F2JarBzC0CqRAETR8SRBMwi0dbh8FvWIX2763s40itxOPRGrN/CXXybI3VTCo04x9t16N0+2W0VtLu9BlYqNviZsIkwGvRzgEB2o2+u6xj/1zSDqxv9QRqkKS21Sb4TcwqNijB='}

data = requests.post('https://account.youku.com/login/confirm.json',data=form_data,headers=header)
print(data.content)

import json
print(json.loads(data.text))
