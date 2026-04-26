import os
import cv2
import sys
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Danh sách 100 ID ảnh đa dạng từ COCO train2017 (Xác thực 100% tồn tại)
COCO_IDS = [
    9, 25, 30, 34, 36, 42, 49, 61, 64, 71, 72, 73, 74, 77, 78, 81, 86, 89, 92, 94, 
    109, 110, 113, 127, 133, 136, 138, 142, 143, 144, 149, 151, 154, 164, 165, 192, 194, 196, 201, 208, 
    241, 247, 250, 257, 260, 263, 283, 294, 307, 308, 309, 312, 315, 321, 322, 326, 328, 332, 338, 349, 
    357, 359, 360, 368, 370, 382, 384, 387, 389, 394, 395, 397, 400, 404, 415, 419, 428, 431, 436, 438, 
    443, 446, 450, 459, 471, 472, 474, 486, 488, 490, 491, 502, 508, 510, 514, 520, 529, 531, 532, 536
]

def prepare_benchmark_images(target_num=100, img_size=None):
    """
    Chuẩn bị danh sách ảnh benchmark thực tế từ COCO dataset.
    """
    # Khởi tạo path để có thể chạy từ bất kỳ đâu
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    cache_dir = os.path.join(project_root, "assets", "benchmark_images")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Kiểm tra và tải ảnh COCO
    base_url = "http://images.cocodataset.org/train2017/"
    
    def download_coco_img(coco_id):
        filename = f"{coco_id:012d}.jpg"
        path = os.path.join(cache_dir, filename)
        if os.path.exists(path):
            return True
            
        url = base_url + filename
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                with open(path, "wb") as f:
                    f.write(resp.content)
                return True
        except Exception as e:
            print(f"⚠️ Lỗi khi tải ảnh COCO {coco_id}: {e}")
        return False

    print(f"🚀 Kiểm tra và tải bộ 100 ảnh COCO cho benchmark...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(download_coco_img, COCO_IDS[:target_num]))
            
    # Load ảnh vào memory
    images = []
    existing_files = [f for f in os.listdir(cache_dir) if f.endswith('.jpg')]
    
    print(f"🖼️ Đang nạp {len(existing_files[:target_num])} ảnh COCO vào bộ nhớ...")
    for f in sorted(existing_files)[:target_num]:
        img = cv2.imread(os.path.join(cache_dir, f))
        if img is not None:
            # Resize nếu cần (Mặc định giữ nguyên gốc vì YOLO tự resize trong detector.py)
            if img_size:
                img = cv2.resize(img, img_size)
            images.append(img)
            
    if not images:
        print("⚠️ Không tải được ảnh COCO, sử dụng ảnh ngẫu nhiên giả lập.")
        images = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)]
        
    return images

class TeeLogger:
    """Ghi log ra cả màn hình terminal và file cùng lúc"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
