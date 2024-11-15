#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/30 10:15                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

import logging
from colorlog import ColoredFormatter

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
    datefmt='%H:%M:%S.%f',
    # datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':'cyan',
        'INFO':     'white,bold',
        'INFOV':    'cyan,bold',
        'WARNING':  'yellow',
        'ERROR':    'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
 )
ch.setFormatter(formatter)

log = logging.getLogger('ssgan')
log.setLevel(logging.DEBUG)
log.handlers=[] # No duplicated handlers
log.propagate=False # workaround for duplicated logs in ipython
log.addHandler(ch)

logging.addLevelName(logging.INFO+1,'INFOV')
def _infov(self,msg,*args,**kwargs):
    self.log(logging.INFO+1,msg,*args,**kwargs)

logging.Logger.infov = _infov

