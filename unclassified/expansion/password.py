# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2026/1/29 14:13                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import secrets
import string
import json
import hashlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import getpass
import pyperclip
import qrcode
from datetime import datetime
import os

class PasswordManager:
    """安全的密码生成与管理器"""

    def __init__(self, master_password=None):
        self.key_file = "password_key.key"
        self.data_file = "passwords.encrypted"
        self.fernet = None

        if master_password:
            self._setup_encryption(master_password)

    def _setup_encryption(self, master_password):
        """设置加密"""
        # 从密码生成密钥
        password_bytes = master_password.encode()
        salt = b'salt_123'  # 实际应用中应使用随机salt

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))

        self.fernet = Fernet(key)

    def generate_password(self, length=16, use_special=True):
        """生成随机密码"""
        characters = string.ascii_letters + string.digits

        if use_special:
            characters += "!@#$%^&*()_+-=[]{}|;:,.<>?"

        while True:
            password = ''.join(secrets.choice(characters) for _ in range(length))

            # 确保密码强度
            if (any(c.islower() for c in password) and
                    any(c.isupper() for c in password) and
                    any(c.isdigit() for c in password)):

                if use_special:
                    if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
                        return password
                else:
                    return password

    def check_password_strength(self, password):
        """检查密码强度"""
        score = 0
        suggestions = []

        # 长度检查
        if len(password) >= 12:
            score += 2
        elif len(password) >= 8:
            score += 1
        else:
            suggestions.append("密码太短，建议至少8个字符")

        # 字符类型检查
        if any(c.islower() for c in password):
            score += 1
        else:
            suggestions.append("添加小写字母")

        if any(c.isupper() for c in password):
            score += 1
        else:
            suggestions.append("添加大写字母")

        if any(c.isdigit() for c in password):
            score += 1
        else:
            suggestions.append("添加数字")

        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if any(c in special_chars for c in password):
            score += 1
        else:
            suggestions.append("添加特殊字符")

        # 评估强度
        if score >= 5:
            strength = "非常强"
        elif score >= 4:
            strength = "强"
        elif score >= 3:
            strength = "中等"
        else:
            strength = "弱"

        return {
            'score': score,
            'strength': strength,
            'suggestions': suggestions
        }

    def save_password(self, service, username, password, notes=""):
        """保存密码"""
        if not self.fernet:
            print("请先设置主密码")
            return False

        try:
            # 读取现有数据
            data = self._load_data() or {}

            # 添加新密码
            entry = {
                'service': service,
                'username': username,
                'password': password,
                'notes': notes,
                'created': datetime.now().isoformat(),
                'updated': datetime.now().isoformat()
            }

            if service in data:
                data[service].append(entry)
            else:
                data[service] = [entry]

            # 加密并保存
            encrypted_data = self.fernet.encrypt(json.dumps(data).encode())
            with open(self.data_file, 'wb') as f:
                f.write(encrypted_data)

            print(f"密码已保存到: {service}")
            return True

        except Exception as e:
            print(f"保存失败: {e}")
            return False

    def _load_data(self):
        """加载加密数据"""
        if not os.path.exists(self.data_file):
            return None

        try:
            with open(self.data_file, 'rb') as f:
                encrypted_data = f.read()

            decrypted_data = self.fernet.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except:
            return None

    def get_password(self, service):
        """获取密码"""
        data = self._load_data()

        if data and service in data:
            return data[service]
        else:
            return None

    def generate_qrcode(self, text, filename="password_qr.png"):
        """生成二维码"""
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(text)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img.save(filename)
        print(f"二维码已保存: {filename}")
        return filename

    def interactive_mode(self):
        """交互模式"""
        print("=== 密码管理器 ===")
        print("1. 生成随机密码")
        print("2. 检查密码强度")
        print("3. 保存密码")
        print("4. 查看密码")
        print("5. 生成密码二维码")
        print("6. 退出")

        while True:
            choice = input("\n请选择操作 (1-6): ").strip()

            if choice == '1':
                length = int(input("密码长度 (默认16): ") or "16")
                use_special = input("包含特殊字符? (y/n, 默认y): ").lower() != 'n'

                password = self.generate_password(length, use_special)
                print(f"生成的密码: {password}")

                copy = input("复制到剪贴板? (y/n): ").lower()
                if copy == 'y':
                    pyperclip.copy(password)
                    print("已复制到剪贴板")

            elif choice == '2':
                password = getpass.getpass("输入要检查的密码: ")
                result = self.check_password_strength(password)

                print(f"\n密码强度: {result['strength']} ({result['score']}/5)")
                if result['suggestions']:
                    print("建议:")
                    for suggestion in result['suggestions']:
                        print(f"  - {suggestion}")

            elif choice == '3':
                service = input("服务/网站名称: ")
                username = input("用户名/邮箱: ")
                password = getpass.getpass("密码 (留空自动生成): ")

                if not password:
                    password = self.generate_password()
                    print(f"已生成密码: {password}")

                notes = input("备注 (可选): ")
                self.save_password(service, username, password, notes)

            elif choice == '4':
                service = input("服务名称: ")
                passwords = self.get_password(service)

                if passwords:
                    for i, entry in enumerate(passwords, 1):
                        print(f"\n[{i}] {entry['service']}")
                        print(f"   用户名: {entry['username']}")
                        print(f"   密码: {'*' * len(entry['password'])}")
                        print(f"   创建时间: {entry['created']}")
                        if entry['notes']:
                            print(f"   备注: {entry['notes']}")
                else:
                    print("未找到相关密码")

            elif choice == '5':
                text = input("输入要编码为二维码的文本: ")
                filename = input("保存文件名 (默认: password_qr.png): ") or "password_qr.png"
                self.generate_qrcode(text, filename)

            elif choice == '6':
                print("退出密码管理器")
                break

            else:
                print("无效选择，请重试")


if __name__ == '__main__':
    mgr = PasswordManager()
    mgr.interactive_mode()