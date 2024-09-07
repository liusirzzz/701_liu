
import numpy as np
import cv2
import torch
from lprr.LPRNet import CHARS, build_lprnet
import numpy as np
from PIL import ImageDraw, Image, ImageFont
import pdb


def transform(img):
    img = img.astype('float32')
    img /= 255
    img = np.transpose(img, (2, 0, 1))
    return img


def de_lpr(coord, im0):
    img = im0[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]
    ims = []
    # cut = cv2.imread(img)
    im = cv2.resize(img, (94, 24))
    im = transform(im)
    ims.append(im)
    ims = torch.Tensor(ims)
    lprnet = build_lprnet(lpr_max_len=8,
                          phase=True,
                          class_num=len(CHARS),
                          dropout_rate=0.5)
    device = torch.device("cuda:0" if True else "cpu")
    lprnet.to(device)
    lprnet.load_state_dict(
        torch.load(r"yolo_lprnet/master_liu/lprr/Final_LPRNet_model.pth"))
    prebs = lprnet(ims.to(device))  # classifier prediction
    prebs = prebs.cpu().detach().numpy()
    preb_labels = list()
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]  # 对每张图片 [68, 18]
        preb_label = list()
        # pdb.set_trace()
        flag = False
        k = 0
        for j in range(preb.shape[1]):  # 18  返回序列中每个位置最大的概率对应的字符idx  其中'-'是67
            if (np.argmax(preb[:, j], axis=0)) >= 0 and (np.argmax(preb[:, j], axis=0)) <= 30:
                if flag==True:
                    preb_label.pop(j - k - 1)
                    k += 1
                else:
                    flag = True
            preb_label.append(np.argmax(preb[:, j], axis=0))
        # pdb.set_trace()
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:  # 记录重复字符
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:  # 去除重复字符和空白字符'-'
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)
    # pdb.set_trace()
    plat_num = np.array(preb_labels)
    # print(plat_num)
    return plat_num


# def dr_plate(im0, coord, plat_num):
#     img = Image.fromarray(im0)
#     draw = ImageDraw.Draw(img)
#     font = ImageFont.truetype("C:\Windows\Fonts\方正粗黑宋简体.ttf", 20)
#     x1 = int(coord[0])
#     x2 = int(coord[1])
#     plate = np.array(plat_num)
#     a = ""
#     for i in range(0, plate.shape[1]):
#         b = CHARS[plate[0][i]]
#         a += b
#     import pdb
#     pdb.set_trace()
#     draw.text((x1, x2), a, (255, 0, 0), font=font)
#     # cv2.putText(im0,
#     #             a, (x1, x2),
#     #             0,
#     #             1, (255, 0, 0),
#     #             thickness=2,
#     #             lineType=cv2.LINE_AA)
#     return np.array(img)

def dr_plate(im0, coord, plat_num):
    x1 = int(coord[0])
    x2 = int(coord[1])

    # 转换车牌号
    plate = np.array(plat_num)
    a = ""
    for i in range(0, plate.shape[1]):
        b = CHARS[plate[0][i]]
        a += b  # 将字符连接成字符串
    
    print("车牌号为：", a)
    # 画框
    cv2.rectangle(im0, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0, 0, 255), 2, cv2.LINE_AA)
    # 绘制文本，支持中文

    # 绘制文本，支持中文
    cv2.rectangle(im0, (int(x1), int(x2) - 30), (int(x1 + 110), int(x2)), (0, 0, 255), -1, cv2.LINE_AA)
    # 加载图像并转换为 PIL 格式
    image = Image.fromarray(im0)
    draw = ImageDraw.Draw(image)
    # 加载字体，设置字体大小
    font = ImageFont.truetype("C:/Windows/Fonts/SimHei.ttf", 25)  # 32 是字体大小，可以调整
    text_color = (255, 255, 255)  # 颜色
    draw.text((x1, x2 - 30), a, font=font, fill=text_color)

    # 将图像转换回 OpenCV 格式
    im0 = np.array(image)
    # cv2.imshow("车牌", im0)
    # cv2.waitKey(0)
    return im0