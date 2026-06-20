"""
Tiện ích quản lý luồng Video đa luồng.
"""

import cv2
import threading
import queue

class VideoStream:
    """
    VideoStream: Luồng đọc Camera đa luồng để đảm bảo hiển thị mượt mà
    và không làm chậm quá trình xử lý AI.
    """
    def __init__(self, src=0, width=1280, height=720, buffer_size=1):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        self.frame_queue = queue.Queue(maxsize=2)
        self.last_frame = None  # Frame gần nhất, dùng fallback khi timeout
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.stopped = True
                break
            
            self.last_frame = frame
            # Xóa frame cũ nếu queue đầy để luôn có ảnh mới nhất
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def read(self, timeout=0.1):
        """Trả về (True, frame) — dùng last_frame làm fallback nếu timeout."""
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return True, frame
        except queue.Empty:
            if self.last_frame is not None:
                return True, self.last_frame
            return False, None

    def stop(self):
        self.stopped = True
        self.stream.release()
