import hyperlpr3 as lpr3
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def drawRectBox(image, rect, addText, fontC, conf):
    """
    车牌识别，绘制矩形框与结果
    :param image: 原始图像
    :param rect: 矩形框坐标
    :param addText:车牌号
    :param fontC: 字体
    :return:
    """
    # 绘制车牌位置方框
    cv2.rectangle(image, (int(round(rect[0])), int(round(rect[1]))),
                 (int(round(rect[2]) + 15), int(round(rect[3]) + 15)),
                 (0, 0, 255), 2)
    # 绘制字体背景框
    cv2.rectangle(image, (int(rect[0] - 1), int(rect[1]) - 10), (int(rect[0] + 80), int(rect[1])), (0, 0, 255), -1, cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    addText = addText + " " + str(round(conf, 2))
    draw.text((int(rect[0]), int(rect[1]-10)), addText, (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex

# 读取选择的图片
input_folder = 'hyperlpr_based\input_img'
output_folder = 'hyperlpr_based\output_img'
catcher = lpr3.LicensePlateCatcher()
for file in os.listdir(input_folder):
    img_path = os.path.join(input_folder, file)
    img = cv2.imread(img_path)

    ouput_path = os.path.join(output_folder, file)

    all_res = catcher(img)
    print(file, all_res[0][0])
    # 车牌标注的字体
    fontC = ImageFont.truetype("C:\Windows\Fonts\方正粗黑宋简体.ttf", 10, 0)
    # all_res为多个车牌信息的列表，取第一个车牌信息
    lisence, conf, _, boxes = all_res[0]
    image = drawRectBox(img, boxes, lisence, fontC, conf)
    cv2.imwrite(ouput_path, image)

    # cv2.imshow('RecognitionResult', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()