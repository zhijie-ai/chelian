# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2026/1/29 16:37                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import tkinter as tk
from tkinter import messagebox
import threading

def show_alert(message="警告！", title="报警"):
    """基本弹窗报警"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.attributes('-topmost', True)  # 窗口置顶
    messagebox.showwarning(title, message)
    root.destroy()

# 在新线程中显示，避免阻塞主程序
alert_thread = threading.Thread(target=show_alert, args=("系统检测到异常！", "系统报警"))
alert_thread.start()