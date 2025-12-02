import urllib.request
import os

print('Downloading YOLOv8n ONNX model...')
print('Source: Hugging Face')
print('Size: ~6.2 MB')
print()

url = 'https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.onnx'
output = 'yolov8n.onnx'

try:
    urllib.request.urlretrieve(url, output)
    size_mb = os.path.getsize(output) / 1024 / 1024
    print(f'✅ Download complete!')
    print(f'📁 File: {output}')
    print(f'📊 Size: {size_mb:.1f} MB')
except Exception as e:
    print(f'❌ Download failed: {e}')
    print()
    print('Please manually download from:')
    print('https://huggingface.co/Ultralytics/YOLOv8/tree/main')
    print('And save as yolov8n.onnx in the GUI folder')
