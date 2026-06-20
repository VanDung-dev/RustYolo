"""
Module cốt lõi quản lý nhận diện khuôn mặt và cơ sở dữ liệu nhân viên.
"""

import pyarrow as pa
import numpy as np
import rust_yolo
import os
import sqlite3
import logging
from config import DB_PATH, DEFAULT_EP

# Cấu hình logger cho module core
logger = logging.getLogger(__name__)

class AttendanceCore:
    """
    AttendanceCore: Engine chính điều phối việc nhận diện khuôn mặt và quản lý database.
    Kết nối giữa Python logic và Rust high-performance engine qua Apache Arrow.
    """
    def __init__(self, detector_path, arcface_path, db_path=DB_PATH, execution_provider=DEFAULT_EP):
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.getcwd(), db_path)
        
        self.db_path = db_path
        self._init_db_python(db_path)
        
        self.known_users = []
        self.load_known_users()

        logger.info(f"-> Đang khởi tạo FaceToolbox (EP: {execution_provider})...")
        self.face_tools = rust_yolo.FaceToolbox()
        
        # Nạp các module AI (Detector, Embedder) từ Rust Engine
        logger.info(f"-> Đang nạp SCRFD Detector: {os.path.basename(detector_path)}...")
        self.face_tools.load_detector(detector_path, (640, 640), execution_provider)
        
        logger.info(f"-> Đang nạp ArcFace model: {os.path.basename(arcface_path)}...")
        self.face_tools.load_embedder(arcface_path, execution_provider)

    @staticmethod
    def _init_db_python(db_path):
        """Khởi tạo cấu trúc bảng SQLite nếu chưa tồn tại"""
        sqls = [
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )"""
        ]
        with sqlite3.connect(db_path) as conn:
            for sql in sqls:
                conn.execute(sql)
            conn.commit()

    def load_known_users(self):
        """Tải tất cả user và embeddings từ DB vào bộ nhớ để so khớp nhanh"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, embedding FROM users")
        rows = cursor.fetchall()
        
        self.known_users = []
        for row in rows:
            embedding = np.frombuffer(row[2], dtype=np.float32)
            self.known_users.append({
                "id": row[0],
                "name": row[1],
                "embedding": embedding
            })
        conn.close()
        logger.info(f"  [DB] Đã tải {len(self.known_users)} người dùng vào bộ nhớ.")

    def process_frame(self, frame, threshold=0.6, max_faces=50):
        """
        Xử lý frame: Phát hiện khuôn mặt -> Trích xuất Embedding -> So khớp danh tính.
        Sử dụng cơ chế Zero-Copy qua Apache Arrow để đạt hiệu năng tối đa.
        max_faces: giới hạn số khuôn mặt xử lý mỗi frame (tránh treo khi đám đông)
        Không cần cv2.cvtColor — Rust tự swap BGR→RGB trong normalize tensor.
        """
        # 1. Phát hiện tất cả khuôn mặt (Gửi ảnh xuống Rust, nhận về Arrow Capsule)
        try:
            arr_cap, sch_cap = self.face_tools.detect_faces_to_arrow(frame, threshold)
            detections_arr = pa.Array._import_from_c_capsule(sch_cap, arr_cap)
            
            if len(detections_arr) == 0:
                return []
            
            num_faces = min(len(detections_arr), max_faces)
            
            # Lấy landmarks - dùng to_pylist (ListArray không support to_numpy trực tiếp)
            all_landmarks = detections_arr.field("landmarks").to_pylist()[:num_faces]
            
            # 2. Thu thập embeddings xử lý hàng loạt (Batch Inference trong Rust)
            e_arr_cap, e_sch_cap = self.face_tools.get_embeddings_batch_to_arrow(frame, all_landmarks)
            emb_flat = pa.Array._import_from_c_capsule(e_sch_cap, e_arr_cap).to_numpy()
            
            # Reshape mảng embeddings phẳng thành (Số mặt, 512 chiều)
            embeddings = emb_flat.reshape(num_faces, 512)
            
            results = []
            
            # 3. So khớp danh tính (Sử dụng Vectorized matching với NumPy để cực nhanh)
            if self.known_users:
                known_embs = np.stack([u['embedding'] for u in self.known_users])
                # Tính Cosine Similarity bằng Matrix Multiplication (N, 512) @ (512, M) -> (N, M)
                similarities = np.dot(embeddings, known_embs.T)
                
                best_indices = np.argmax(similarities, axis=1)
                best_scores = np.max(similarities, axis=1)
            else:
                best_indices = [0] * num_faces
                best_scores = [-1.0] * num_faces

            # Gom kết quả cuối cùng cho từng khuôn mặt
            det_x1 = detections_arr.field("x1").to_numpy(zero_copy_only=False)[:num_faces]
            det_y1 = detections_arr.field("y1").to_numpy(zero_copy_only=False)[:num_faces]
            det_x2 = detections_arr.field("x2").to_numpy(zero_copy_only=False)[:num_faces]
            det_y2 = detections_arr.field("y2").to_numpy(zero_copy_only=False)[:num_faces]
            
            for i in range(num_faces):
                score = float(best_scores[i])
                is_known = score > 0.45
                
                # Ánh xạ thông tin user (Default là Unknown nếu không khớp)
                user = self.known_users[int(best_indices[i])] if is_known else {"name": "Unknown", "id": None}
                
                results.append({
                    "bbox": [float(det_x1[i]), float(det_y1[i]), float(det_x2[i]), float(det_y2[i])],
                    "identity": user['name'],
                    "confidence": score,
                    "user_id": user['id']
                })
                
            return results
        except Exception as e:
            logger.error(f"  [LỖI] Xử lý frame (Arrow Pipeline): {e}", exc_info=True)
            return []

    def get_face_embedding(self, face_rgb_112):
        """Trích xuất embedding trực tiếp từ một ảnh khuôn mặt đã được chuẩn hóa (112x112)"""
        data = face_rgb_112.tobytes() if hasattr(face_rgb_112, "tobytes") else face_rgb_112
        emb_list = self.face_tools.get_embedding(data)
        return np.array(emb_list, dtype=np.float32)

    def add_user(self, name, embedding):
        """Đăng ký nhân viên mới vào cơ sở dữ liệu"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            cursor.execute("INSERT INTO users (name, embedding) VALUES (?, ?)", (name, embedding_bytes))
            new_id = cursor.lastrowid
            conn.commit()
            conn.close()
            # Cập nhật lại danh sách trong bộ nhớ
            self.load_known_users()
            return new_id
        except Exception as e:
            logger.error(f"  [LỖI DB] Không thể thêm người dùng: {e}")
            return None

    def log_attendance(self, user_id):
        """Ghi nhận thời gian điểm danh vào nhật ký"""
        if user_id is None:
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO attendance_logs (user_id) VALUES (?)", (user_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"  [LỖI DB] Không thể ghi nhật ký điểm danh: {e}")
