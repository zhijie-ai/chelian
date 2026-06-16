# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2026/1/29 14:29                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import requests
import json
import socket
import dns.resolver
import whois
from urllib.parse import urlparse
import ssl
import datetime
import concurrent.futures
import time

class NetworkTools:
    """网络工具集"""

    @staticmethod
    def check_website_status(url):
        """检查网站状态"""
        try:
            response = requests.get(url, timeout=10)
            return {
                'url': url,
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'headers': dict(response.headers),
                'size': len(response.content)
            }
        except requests.exceptions.RequestException as e:
            return {'url': url, 'error': str(e)}

    @staticmethod
    def batch_check_urls(urls, max_workers=10):
        """批量检查URL"""
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(NetworkTools.check_website_status, url): url
                             for url in urls}

            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({'url': url, 'error': str(e)})

        return results

    @staticmethod
    def get_domain_info(domain):
        """获取域名信息"""
        info = {'domain': domain}

        try:
            # DNS查询
            answers = dns.resolver.resolve(domain, 'A')
            info['ip_addresses'] = [str(rdata) for rdata in answers]

            # WHOIS查询
            whois_info = whois.whois(domain)
            info['whois'] = {
                'registrar': whois_info.registrar,
                'creation_date': str(whois_info.creation_date),
                'expiration_date': str(whois_info.expiration_date),
                'name_servers': whois_info.name_servers
            }

            # SSL证书信息
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()

                    # 证书过期时间
                    expire_date = datetime.datetime.strptime(cert['notAfter'],
                                                             '%b %d %H:%M:%S %Y %Z')
                    days_left = (expire_date - datetime.datetime.now()).days

                    info['ssl'] = {
                        'issuer': dict(x[0] for x in cert['issuer']),
                        'subject': dict(x[0] for x in cert['subject']),
                        'expires': str(expire_date),
                        'days_remaining': days_left
                    }

        except Exception as e:
            info['error'] = str(e)

        return info

    @staticmethod
    def port_scanner(host, start_port=1, end_port=1024, timeout=1):
        """端口扫描器"""
        open_ports = []

        def check_port(port):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(timeout)
                    result = sock.connect_ex((host, port))
                    if result == 0:
                        return port
            except:
                pass
            return None

        print(f"扫描 {host} 端口 {start_port}-{end_port}...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = {executor.submit(check_port, port): port
                       for port in range(start_port, end_port + 1)}

            for future in concurrent.futures.as_completed(futures):
                port = futures[future]
                try:
                    result = future.result()
                    if result:
                        open_ports.append(result)
                        print(f"端口 {result} 开放")
                except:
                    pass

        return sorted(open_ports)

    @staticmethod
    def download_file(url, save_path=None, chunk_size=8192):
        """下载文件"""
        if not save_path:
            save_path = url.split('/')[-1]

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # 显示进度
                        if total_size:
                            percent = (downloaded / total_size) * 100
                            print(f"\r下载进度: {percent:.1f}%", end='')

            print(f"\n文件已保存: {save_path}")
            return save_path

        except Exception as e:
            print(f"下载失败: {e}")
            return None

    @staticmethod
    def speed_test(iterations=3):
        """网络速度测试"""
        test_files = [
            'http://ipv4.download.thinkbroadband.com/5MB.zip',
            'http://ipv4.download.thinkbroadband.com/10MB.zip',
            'http://ipv4.download.thinkbroadband.com/20MB.zip'
        ]

        results = []

        for i, url in enumerate(test_files):
            print(f"\n测试 {i+1}/{len(test_files)}: {url}")

            speeds = []
            for j in range(iterations):
                print(f"  第 {j+1} 次测试...")

                start_time = time.time()
                response = requests.get(url, stream=True)

                # 读取前1MB数据来测试速度
                size = 0
                for chunk in response.iter_content(chunk_size=1024):
                    size += len(chunk)
                    if size >= 1024 * 1024:  # 1MB
                        break

                elapsed = time.time() - start_time
                speed = (size / 1024 / 1024) / elapsed  # MB/s
                speeds.append(speed)

                print(f"    速度: {speed:.2f} MB/s")

            avg_speed = sum(speeds) / len(speeds)
            results.append({
                'file_size': '1MB',
                'speeds': speeds,
                'average': avg_speed
            })

        return results


if __name__ == '__main__':
    net = NetworkTools()
    print(net.port_scanner('localhost'))