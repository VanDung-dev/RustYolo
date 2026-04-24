"""
Module Video Stream - Luồng đọc Camera tốc độ cao

Sử dụng Buffer riêng để không làm nghẽn luồng xử lý AI.
Hỗ trợ cả camera local và stream URL (rtsp://, http://, tcp://).
"""

import threading
import queue
import logging

import cv2

logger = logging.getLogger(__name__)


class VideoStream:
    """Threaded video capture stream to handle I/O without blocking
    
    Args:
        src: Camera ID (int) hoặc URL stream (str) như 'rtsp://', 'http://', 'tcp://'
        width: Độ phân giải rộng (chỉ áp dụng cho camera local)
        height: Độ phân giải cao (chỉ áp dụng cho camera local)
        fps: FPS mục tiêu (chỉ áp dụng cho camera local)
    """
    def __init__(self, src=0, width=1920, height=1080, fps=60):
        # Xác định nếu src là URL string hay camera ID integer
        self.is_url = isinstance(src, str)
        
        # Nếu là URL, không áp dụng các cấu hình camera (không có ý nghĩa với stream)
        if self.is_url:
            self.stream = cv2.VideoCapture(src)
            logger.info(f"Đang kết nối đến stream URL: {src}")
        else:
            self.stream = cv2.VideoCapture(src)
            
            # Cấu hình camera độ phân giải cao và fps từ config
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.stream.set(cv2.CAP_PROP_FPS, fps)
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

        self.frame_queue = queue.Queue(maxsize=2)
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            (grabbed, frame) = self.stream.read()
            if grabbed:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)

    def read(self):
        try:
            # Thu thập frame mới nhất, timeout ngắn để không block main thread
            return True, self.frame_queue.get(timeout=0.1)
        except queue.Empty:
            return False, None

    def stop(self):
        self.stopped = True
        self.stream.release()
