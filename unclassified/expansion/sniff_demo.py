# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2026/1/29 15:26                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
from scapy.all import sniff
import time
from collections import defaultdict

class TrafficAnalyzer:
    def __init__(self):
        self.traffic_stats = defaultdict(lambda: {'bytes': 0, 'packets': 0})
        self.start_time = time.time()

    def packet_handler(self, packet):
        """处理每个数据包"""
        if packet.haslayer('IP'):
            src = packet['IP'].src
            dst = packet['IP'].dst
            length = len(packet)

            # 统计源IP
            self.traffic_stats[src]['bytes'] += length
            self.traffic_stats[src]['packets'] += 1

            # 统计目标IP
            self.traffic_stats[dst]['bytes'] += length
            self.traffic_stats[dst]['packets'] += 1

    def start_monitoring(self, timeout=30):
        """开始监控网络流量"""
        print("开始监控网络流量...")
        sniff(prn=self.packet_handler, timeout=timeout)
        self.display_stats()

    def display_stats(self):
        """显示统计信息"""
        duration = time.time() - self.start_time
        print(f"\n监控时长: {duration:.2f} 秒")
        print("="*60)

        sorted_stats = sorted(
            self.traffic_stats.items(),
            key=lambda x: x[1]['bytes'],
            reverse=True
        )

        for ip, stats in sorted_stats[:10]:  # 显示前10个
            bps = stats['bytes'] / duration
            print(f"IP: {ip}")
            print(f"  总字节: {stats['bytes']} | 总包数: {stats['packets']}")
            print(f"  平均速率: {bps:.2f} B/s ({bps/1024:.2f} KB/s)")
            print("-"*40)

# 使用示例
if __name__ == "__main__":
    analyzer = TrafficAnalyzer()
    analyzer.start_monitoring(timeout=10)