# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2026/1/28 18:18                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import pyautogui
import pytesseract
from PIL import Image, ImageGrab, ImageEnhance
import numpy as np
import cv2
import keyboard
import time
import pyperclip
import json
from datetime import datetime
import os

class ScreenshotOCR:
    """屏幕截图OCR识别工具"""

    def __init__(self):
        self.screenshot_dir = "../screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def take_screenshot(self, region=None, save=True):
        """截图"""
        if region:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.screenshot_dir}/screenshot_{timestamp}.png"
            screenshot.save(filename)
            print(f"截图已保存: {filename}")

        return screenshot

    def preprocess_image(self, image):
        """图像预处理以提高OCR准确率"""
        # 转换为灰度图
        if image.mode != 'L':
            image = image.convert('L')

        # 转换为numpy数组
        img_array = np.array(image)

        # 使用OpenCV进行图像处理
        # 二值化
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 去噪
        denoised = cv2.medianBlur(binary, 3)

        # 返回处理后的图像
        return Image.fromarray(denoised)

    def ocr_from_screenshot(self, region=None, preprocess=True):
        """从截图进行OCR识别"""
        # 截图
        screenshot = self.take_screenshot(region, save=True)

        # 图像预处理
        if preprocess:
            processed = self.preprocess_image(screenshot)
        else:
            processed = screenshot

        # OCR识别
        try:
            text = pytesseract.image_to_string(processed, lang='chi_sim+eng')
            return text.strip()
        except Exception as e:
            print(f"OCR识别失败: {e}")
            return ""

    def interactive_ocr(self):
        """交互式OCR工具"""
        print("=== 交互式OCR工具 ===")
        print("1. 按 F1: 全屏截图并识别")
        print("2. 按 F2: 选择区域截图并识别")
        print("3. 按 F3: 复制上一次识别结果到剪贴板")
        print("4. 按 ESC: 退出")
        print("-" * 30)

        last_result = ""

        while True:
            if keyboard.is_pressed('f1'):
                print("\n正在全屏截图并识别...")
                result = self.ocr_from_screenshot()
                last_result = result
                print(f"识别结果:\n{result}")
                time.sleep(0.5)  # 防止重复触发

            elif keyboard.is_pressed('f2'):
                print("\n请选择截图区域...")
                time.sleep(0.5)

                # 使用pyautogui的locateOnScreen等待用户选择（简化版）
                # 实际应用中可能需要更复杂的区域选择机制
                region = (100, 100, 400, 300)  # 示例区域
                print(f"使用示例区域: {region}")

                result = self.ocr_from_screenshot(region)
                last_result = result
                print(f"识别结果:\n{result}")
                time.sleep(0.5)

            elif keyboard.is_pressed('f3') and last_result:
                pyperclip.copy(last_result)
                print("\n上次识别结果已复制到剪贴板")
                time.sleep(0.5)

            elif keyboard.is_pressed('esc'):
                print("\n退出OCR工具")
                break

    def batch_ocr_images(self, image_folder):
        """批量处理图片文件夹进行OCR"""
        results = []

        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                filepath = os.path.join(image_folder, filename)
                print(f"处理: {filename}")

                try:
                    image = Image.open(filepath)
                    processed = self.preprocess_image(image)
                    text = pytesseract.image_to_string(processed, lang='chi_sim+eng')

                    results.append({
                        'file': filename,
                        'text': text.strip(),
                        'timestamp': datetime.now().isoformat()
                    })

                    print(f"  识别完成: {len(text)} 字符")

                except Exception as e:
                    print(f"  处理失败: {e}")

        # 保存结果
        output_file = f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n批量处理完成！结果已保存到: {output_file}")
        return results

# 安装所需库
# pip install pyautogui pillow pytesseract opencv-python numpy keyboard pyperclip

if __name__ == '__main__':
    ocr = ScreenshotOCR()
    ocr.interactive_ocr()