# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2026/1/29 16:38                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import tkinter as tk
import threading
import time

class BlinkingAlert:
    def __init__(self, message="警告！", title="报警"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.attributes('-topmost', True)  # 置顶

        # 设置窗口样式
        self.root.configure(bg='red')
        self.label = tk.Label(
            self.root,
            text=message,
            font=("Arial", 24, "bold"),
            bg='red',
            fg='white'
        )
        self.label.pack(padx=50, pady=30)

        self.is_blinking = True
        self.blink_interval = 500  # 毫秒

    def blink(self):
        """窗口闪烁效果"""
        current_color = self.root.cget("bg")
        new_color = 'yellow' if current_color == 'red' else 'red'
        self.root.configure(bg=new_color)
        self.label.configure(bg=new_color)

        if self.is_blinking:
            self.root.after(self.blink_interval, self.blink)

    def run(self):
        self.root.after(0, self.blink)
        self.root.mainloop()

    def stop(self):
        self.is_blinking = False
        self.root.destroy()

# 使用示例
alert = BlinkingAlert("高温报警！温度超过阈值", "温度监控")
alert.run()
# thread = threading.Thread(target=alert.run)
# thread.start()

# 5秒后停止报警
# time.sleep(5)
# alert.stop()