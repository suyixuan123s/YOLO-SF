import matplotlib.pyplot as plt
# @yoloair 🥭
import numpy as np


x1 = np.array([0.6, 1.2, 1.7, 2.7, 4.8])
y1 = np.array([45.7, 56.8, 64.1, 67.3, 68.9])

# YOLOv5
x2 = np.array([0.6, 1.2, 1.7, 2.7])
y2 = np.array([37.4, 45.4, 49.0, 50.7])

# YOLOv6
x3 = np.array([0.9, 1.7, 2.2, 3.6])
y3 = np.array([37.5, 45.0, 50.0, 52.8])

# YOLOv7
x4 = np.array([1.3, 2.1, 3.7])
y4 = np.array([37.4, 43.2, 51.2])

# YOLOv8
x5 = np.array([0.9, 1.3, 2.7, 3.9])
y5 = np.array([37.3, 44.9, 50.2, 53.9])

# PPYOLOE
x6 = np.array([1.5, 1.9, 2.8, 3.9])
y6 = np.array([37.5, 45.9, 50.8, 51.3])

# DAMOYOLO
x7 = np.array([1.6, 2.3, 2.9, 3.9])
y7 = np.array([37.9, 46.9, 51.8, 52.7])

# YOLOX
x8 = np.array([1.1, 1.7, 2.7, 3.9])
y8 = np.array([37.5, 45.3, 50.3, 50.9])

# MS-YOLO
x = np.array([0.4, 1.1, 1.7, 2.3])
y = np.array([39.5, 46.0, 50.9, 52.9])

# 以上数据按自己的需求进行修改

plt.grid(True, linestyle="-", alpha=0.5)
plt.plot(x2, y2, label='YOLOv5', marker = 's', linewidth =1.0, color='green')
plt.plot(x3, y3, label='YOLOv6', marker = '+',linewidth =1.0, color='blue')
plt.plot(x4, y4, label='YOLOv7', marker = 's',linewidth =1.0, color='pink')
plt.plot(x5, y5, label='YOLOv8', marker = '+',linewidth =1.0, color='c')
plt.plot(x6, y6, label='PP-YOLOE', marker = 's',linewidth =1.0, color='orange')
plt.plot(x7, y7, label='DAMOYOLO', marker = '+',linewidth =1.0, color='cornflowerblue')
plt.plot(x8, y8, label='YOLOX', marker = 's',linewidth =1.0, color='c')
plt.plot(x, y, label='Our(YOLOAir)', marker = 'o',linewidth =2.0, color='red')

plt.text(x[0],y[0], 'Our-S',c='black')
plt.text(x[1],y[1], 'Our-M',c='black')
plt.text(x[2],y[2], 'Our-L',c='black')
plt.text(x[3],y[3], 'Our-X',c='black')

plt.ylabel('mAP0.5-0.95(%)val')
plt.xlabel('GPU TensorRT FP16 Latency(ms)')
plt.title('Object Detection(mock)')
plt.legend(loc='lower right')

plt.title("YOLOAir",loc = "left",
          size = 15,color = "darkorange",
          style = "oblique",
          family = "Comic Sans MS")
# 图片路径
plt.savefig("F:/YOLO/yolov7-main/yolo/YOLOv7-Pytorch-Segmentation/models/yolo_merge_map_performance.png")
#F:\YOLO\yolov7-main\yolo\YOLOv7-Pytorch-Segmentation\models\yolo_merge_map_performance.png