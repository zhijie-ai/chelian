# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2026/1/28 18:05                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import psutil
import platform
import socket
import GPUtil
from datetime import datetime
import json

class SystemMonitor:
    """系统监控器"""

    @staticmethod
    def get_all_info():
        """获取所有系统信息"""
        info = {
            'timestamp': datetime.now().isoformat(),
            'system': SystemMonitor.get_system_info(),
            'cpu': SystemMonitor.get_cpu_info(),
            'memory': SystemMonitor.get_memory_info(),
            'disk': SystemMonitor.get_disk_info(),
            'network': SystemMonitor.get_network_info(),
            'gpu': SystemMonitor.get_gpu_info() if SystemMonitor._has_gpu() else None,
            'processes': SystemMonitor.get_top_processes(5),
            'sensors': SystemMonitor.get_sensors_info()
        }
        return info

    @staticmethod
    def get_system_info():
        """获取系统信息"""
        return {
            'system': platform.system(),
            'node_name': platform.node(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'boot_time': datetime.fromtimestamp(psutil.boot_time()).strftime('%Y-%m-%d %H:%M:%S')
        }

    @staticmethod
    def get_cpu_info():
        """获取CPU信息"""
        return {
            'physical_cores': psutil.cpu_count(logical=False),
            'total_cores': psutil.cpu_count(logical=True),
            'current_usage': psutil.cpu_percent(interval=1, percpu=True),
            'avg_usage': psutil.cpu_percent(interval=1),
            'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }

    @staticmethod
    def get_memory_info():
        """获取内存信息"""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            'total': f"{mem.total / (1024**3):.2f} GB",
            'available': f"{mem.available / (1024**3):.2f} GB",
            'used': f"{mem.used / (1024**3):.2f} GB",
            'percent': mem.percent,
            'swap_total': f"{swap.total / (1024**3):.2f} GB",
            'swap_used': f"{swap.used / (1024**3):.2f} GB",
            'swap_percent': swap.percent
        }

    @staticmethod
    def get_disk_info():
        """获取磁盘信息"""
        disks = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disks.append({
                    'device': partition.device,
                    'mountpoint': partition.mountpoint,
                    'fstype': partition.fstype,
                    'total': f"{usage.total / (1024**3):.2f} GB",
                    'used': f"{usage.used / (1024**3):.2f} GB",
                    'free': f"{usage.free / (1024**3):.2f} GB",
                    'percent': usage.percent
                })
            except:
                continue
        return disks

    @staticmethod
    def get_network_info():
        """获取网络信息"""
        net_io = psutil.net_io_counters()
        addrs = psutil.net_if_addrs()

        interfaces = {}
        for interface, snics in addrs.items():
            for snic in snics:
                if snic.family == socket.AF_INET:  # IPv4
                    interfaces[interface] = {
                        'ip_address': snic.address,
                        'netmask': snic.netmask
                    }

        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'interfaces': interfaces
        }

    @staticmethod
    def _has_gpu():
        """检查是否有GPU"""
        try:
            GPUtil.getGPUs()
            return True
        except:
            return False

    @staticmethod
    def get_gpu_info():
        """获取GPU信息"""
        try:
            gpus = GPUtil.getGPUs()
            return [{
                'id': gpu.id,
                'name': gpu.name,
                'load': f"{gpu.load*100:.1f}%",
                'memory_total': f"{gpu.memoryTotal} MB",
                'memory_used': f"{gpu.memoryUsed} MB",
                'memory_free': f"{gpu.memoryFree} MB",
                'temperature': f"{gpu.temperature} °C"
            } for gpu in gpus]
        except:
            return None

    @staticmethod
    def get_top_processes(n=5):
        """获取占用资源最多的进程"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # 按CPU使用率排序
        cpu_sorted = sorted(processes,
                            key=lambda x: x['cpu_percent'] if x['cpu_percent'] else 0,
                            reverse=True)[:n]

        # 按内存使用率排序
        mem_sorted = sorted(processes,
                            key=lambda x: x['memory_percent'] if x['memory_percent'] else 0,
                            reverse=True)[:n]

        return {
            'by_cpu': cpu_sorted,
            'by_memory': mem_sorted
        }

    @staticmethod
    def get_sensors_info():
        """获取传感器信息"""
        try:
            sensors = {
                'temperatures': psutil.sensors_temperatures(),
                'fans': psutil.sensors_fans(),
                'battery': psutil.sensors_battery()._asdict() if psutil.sensors_battery() else None
            }
            return sensors
        except:
            return None

    @staticmethod
    def monitor_loop(interval=5, duration=60):
        """持续监控一段时间"""
        import time

        end_time = time.time() + duration
        history = []

        print("开始系统监控...")
        print("-" * 50)

        while time.time() < end_time:
            info = SystemMonitor.get_all_info()
            history.append(info)

            print(f"\n时间: {info['timestamp']}")
            print(f"CPU使用率: {info['cpu']['avg_usage']:.1f}%")
            print(f"内存使用率: {info['memory']['percent']:.1f}%")
            print(f'{info["sensors"]}')

            if info['gpu']:
                for gpu in info['gpu']:
                    print(f"GPU {gpu['id']}: {gpu['load']} (温度: {gpu['temperature']})")

            time.sleep(interval)

        # 保存监控数据
        with open('../system_monitor_log.json', 'w') as f:
            json.dump(history, f, indent=2)

        print(f"\n监控结束，数据已保存到 system_monitor_log.json")
        return history


if __name__ == '__main__':
    monitor = SystemMonitor()
    # monitor.monitor_loop()
    print(monitor.get_memory_info())