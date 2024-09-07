## 车牌识别系统

### 实现过程

#### hyperlpr_based
- 基于opencv库实现图片与视频文件的读取与保存
- 基于hyperlpr库实现车牌检测功能
- 基于PIL库框出车牌并标签

#### YOLO_lpr_based
- 导入基于pytorch的lprnet库和YOLOv5库及其相应训练权重
- 编写de_lpr函数实现基于lprnet的车牌字符识别功能
- 编写dr_plate函数实现基于PIL库的车牌标签功能
- 修改YOLOv5的detect.py文件，将YOLOv5识别的车牌位置信息传入de_lpr函数识别车牌字符并用dr_plate函数替换原标签函数
- 更改YOLOv5文件读取及模型权重载入的路径

### 运行方法
**环境要求**
- python3.6+
- opencv-python
- hyperlpr
- pytorch
- torchvision
- Pillow
- yolov5


**安装方法**
```bash
pip install opencv-python
pip install hyperlpr
pip install torch torchvision
pip install Pillow
```

**运行hyperlpr_based模型**
```bash
cd hyperlpr_based
python process_img.py
python process_video.py
```
**运行YOLO_lpr_based模型**
```bash
cd yolo_lprnet\master_liu
python detect.py
```
更改检测对象可以在detect.py文件中修改parse_opt中的source参数
### 参考文献

- [LPRNET](https://arxiv.org/abs/1806.10447)
