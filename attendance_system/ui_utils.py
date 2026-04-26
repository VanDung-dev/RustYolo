"""
Tiện ích hỗ trợ vẽ giao diện Tiếng Việt và xử lý font chữ Unicode.
"""

import cv2
import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont
from config import FONT_PATH

# Cấu hình logger cho module
logger = logging.getLogger(__name__)

# Bộ nhớ đệm (Cache) cho các kích thước font đã load để tối ưu hiệu năng
_font_cache = {}

def get_font(size):
    """Tải font từ đĩa hoặc lấy từ cache"""
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
        except Exception as e:
            logger.warning(f"  [CẢNH BÁO] Không thể nạp font tại {FONT_PATH}: {e}. Đang dùng font mặc định.")
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

def draw_text(img, text, pos, font_size=20, color=(0, 255, 0), is_bold=True):
    """
    Vẽ văn bản (hỗ trợ Tiếng Việt) lên ảnh OpenCV.
    
    Args:
        img: Ảnh OpenCV (numpy array định dạng BGR).
        text: Nội dung văn bản cần vẽ.
        pos: Tọa độ (x, y) bắt đầu vẽ (góc trên bên trái).
        font_size: Kích thước chữ.
        color: Màu sắc định dạng BGR (OpenCV style).
        is_bold: Nếu True, sẽ vẽ chữ in đậm bằng cách thêm stroke.
    """
    # 1. Chuyển đổi ảnh OpenCV (BGR) sang PIL (RGB)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # 2. Lấy đối tượng font tương ứng
    font = get_font(font_size)
    
    # 3. Chuyển màu BGR (OpenCV) sang RGB (PIL)
    color_rgb = (color[2], color[1], color[0])
    
    # 4. Vẽ văn bản lên ảnh PIL (Sử dụng stroke_width để giả lập in đậm)
    stroke_width = 1 if is_bold else 0
    draw.text(pos, text, font=font, fill=color_rgb, stroke_width=stroke_width, stroke_fill=color_rgb)
    
    # 5. Chuyển đổi ảnh PIL ngược lại thành OpenCV (BGR)
    res_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # 6. Cập nhật lại mảng byte của ảnh gốc
    img[:] = res_img[:]
    return img
