# YOLOv8 Wildlife Detection

A minimal YOLOv8 wildlife detection system with a beautiful web UI.

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Web UI

```bash
python app.py
```

然后访问 http://localhost:5000

## ✨ 功能

- ✅ 上传图像检测野生动物
- ✅ 实时摄像头检测
- ✅ 视频文件检测
- ✅ 结果可视化（标注检测框）
- ✅ 漂亮的现代化 Web 界面

## 📁 项目结构

```
yolov8-wildlife/
├── app.py                 # Flask 主应用
├── requirements.txt       # 依赖列表
├── models/               # 模型存储目录（自动创建）
├── uploads/              # 用户上传文件目录
├── templates/
│   └── index.html        # Web UI 前端
├── static/              # CSS/JS 静态文件
└── README.md
```

## 🤖 使用的模型

- **YOLOv8 Small (yolov8s.pt)**
- 官方预训练模型（COCO 数据集）
- 80 个类别，包括各种动物
- 首次运行时自动下载

### 支持的动物类别

狗、猫、鸟、马、羊、牛、大象、熊、斑马、长颈鹿、狮子、老虎、猴子、兔子、松鼠、鹿、鸭、鹅、猫头鹰、企鹅等

## 📋 要求

- Python 3.8+
- GPU (推荐，但 CPU 也可以工作)
- 4GB+ 内存

## 🔧 配置

在 `app.py` 中修改以下参数：

```python
UPLOAD_FOLDER = 'uploads'      # 上传文件夹
MODEL_FOLDER = 'models'        # 模型文件夹
MAX_FILE_SIZE = 100 * 1024 * 1024  # 最大文件大小（100MB）
```

## 📚 API 端点

### POST `/api/detect`
上传图像进行检测

**请求：**
```
Content-Type: multipart/form-data
- file: <image_file>
```

**响应：**
```json
{
  "success": true,
  "detections": [
    {
      "class": "dog",
      "confidence": 0.92,
      "bbox": [100, 150, 300, 450]
    }
  ],
  "detection_count": 1,
  "result_image": "/uploads/result_xxx.jpg",
  "original_image": "/uploads/xxx.jpg"
}
```

### GET `/api/classes`
获取模型支持的所有类别

## 🎓 示例使用

### Python 脚本

```python
from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO('yolov8s.pt')

# 检测图像
results = model.predict('wildlife.jpg')

# 显示结果
for result in results:
    result.show()
```

### 实时摄像头检测

```python
import cv2
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow('Detection', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 🛠️ 故障排查

### 问题：模型下载很慢
**解决：** 可以手动从 Ultralytics 下载预训练模型

### 问题：GPU 显存不足
**解决：** 使用更小的模型
```python
model = YOLO('yolov8n.pt')  # nano 版本，更小更快
```

### 问题：检测精度不高
**解决：** 考虑微调模型或使用自定义训练的模型

## 📖 参考资源

- [Ultralytics YOLOv8 官方文档](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [COCO 数据集](https://cocodataset.org/)

## 📄 许可证

MIT License

## 👨‍💻 作者

YOLOv8 Wildlife Detection Project
