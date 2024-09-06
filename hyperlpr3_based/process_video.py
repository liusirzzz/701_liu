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
cap = cv2.VideoCapture('hyperlpr3_based/video1.mp4')
if not cap.isOpened():
    print("Error opening video stream or file")

# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# encoder = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.mp4', encoder, fps, (width, height))
catcher = lpr3.LicensePlateCatcher()

while True:
    bool, frame = cap.read()
    if not bool:
        print("fetch end")
        break
    
    all_res = catcher(frame)
    if all_res:
        # 车牌标注的字体
        for res in all_res:

            fontC = ImageFont.truetype("C:\Windows\Fonts\方正粗黑宋简体.ttf", 10, 0)
            # all_res为多个车牌信息的列表，取第一个车牌信息
            lisence, conf, _, boxes = res
            frame = drawRectBox(frame, boxes, lisence, fontC, conf)

    # out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# out.release()
cap.release()
cv2.destroyAllWindows()