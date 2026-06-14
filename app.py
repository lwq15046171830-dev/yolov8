import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# 创建必要的文件夹
for folder in [UPLOAD_FOLDER, MODEL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# 全局模型
model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """加载 YOLOv8 模型"""
    global model
    if model is None:
        print("Loading YOLOv8 model...")
        model = YOLO('yolov8s.pt')  # 使用官方预训练模型
        print("Model loaded successfully!")
    return model

def annotate_image(image, results):
    """在图像上绘制检测结果"""
    annotated = image.copy()
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            # 获取坐标
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            cls = int(box.cls[0])
            cls_name = result.names[cls]
            
            # 绘制框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    """处理图像检测请求"""
    try:
        load_model()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # 保存上传的文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 读取图像
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Failed to read image'}), 400
        
        # 运行检测
        results = model(image)
        
        # 获取检测结果信息
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = result.names[cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })
        
        # 绘制结果
        annotated = annotate_image(image, results)
        
        # 保存注释后的图像
        output_filename = 'result_' + filename
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_filepath, annotated)
        
        return jsonify({
            'success': True,
            'detections': detections,
            'detection_count': len(detections),
            'result_image': f'/uploads/{output_filename}',
            'original_image': f'/uploads/{filename}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/webcam', methods=['POST'])
def webcam():
    """处理摄像头检测请求"""
    try:
        load_model()
        
        # 这是一个占位符 - 实际的摄像头检测通常需要 WebSocket
        # 为了演示，我们返回一个提示
        return jsonify({
            'info': 'Webcam detection requires WebSocket or streaming endpoint'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """提供上传的文件"""
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename)))
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/classes')
def get_classes():
    """获取模型支持的类别"""
    load_model()
    classes = model.names
    # 过滤出动物相关的类别
    animal_classes = [
        'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant',
        'bear', 'zebra', 'giraffe', 'lion', 'tiger', 'monkey', 'rabbit',
        'squirrel', 'deer', 'duck', 'goose', 'owl', 'penguin'
    ]
    detected_animals = [c for c in animal_classes if c in classes.values()]
    return jsonify({
        'total_classes': len(classes),
        'all_classes': list(classes.values()),
        'animal_classes': detected_animals
    })

if __name__ == '__main__':
    print("🚀 Starting YOLOv8 Wildlife Detection Server...")
    print("📍 Open http://localhost:5000 in your browser")
    print("⏹️  Press Ctrl+C to stop the server")
    app.run(debug=True, host='0.0.0.0', port=5000)
