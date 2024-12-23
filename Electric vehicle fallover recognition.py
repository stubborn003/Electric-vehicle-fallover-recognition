import cv2
from ultralytics import YOLO  # 导入YOLO类
import pyttsx3  # 导入pyttsx3库来进行语音合成

# 加载YOLOv8模型
model = YOLO('best.pt')

def detect_electric_bikes(image):
    results = model(image)  # 执行推理
    detections = results[0].boxes  # 获取检测框
    normal_count = 0
    fallen_count = 0

    for box in detections:
        conf = box.conf[0].item()  # 获取置信度
        cls = int(box.cls[0].item())  # 获取类别

        if conf > 0.5:  # 置信度阈值
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取边界框坐标
            # 绘制边界框和标签
            label = f'up {conf:.2f}' if cls == 0 else f'down {conf:.2f}'
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # 正常用绿色表示，倒伏用红色表示
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # 绘制矩形
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 绘制标签

            if cls == 0:  # 假设0代表正常电动车
                normal_count += 1
            elif cls == 1:  # 假设1代表倒伏电动车
                fallen_count += 1

    return normal_count, fallen_count  # 返回正常和倒伏电动车的数量

image_path = 'c1284062ae089a214772f31eec03423.jpg'  # 设置图片路径
image = cv2.imread(image_path)  # 读取图像

normal_bikes, fallen_bikes = detect_electric_bikes(image)  # 调用检测函数

print(f'正常电动车数量: {normal_bikes}, 倒伏电动车数量: {fallen_bikes}')

# 使用pyttsx3进行语音输出
engine = pyttsx3.init()
engine.say(f'正常电动车数量是 {normal_bikes} 倒伏电动车数量是 {fallen_bikes}')
engine.runAndWait()

# 将图像缩小显示
resize_scale = 0.3  # 设置缩小比例
small_image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)

cv2.imshow('Electric Bike Detection', small_image)  # 显示检测图像
cv2.waitKey(0)  # 等待按键关闭窗口
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口