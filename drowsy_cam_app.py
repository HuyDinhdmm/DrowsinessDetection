import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
import math

# ==== Các hàm và class từ drowsy_detection.py ====
def get_mediapipe_app(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return face_mesh

def distance(point_1, point_2):
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist

def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
    except:
        ear = 0.0
        coords_points = None
    return ear, coords_points

def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    Avg_EAR = (left_ear + right_ear) / 2.0
    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)

def plot_eye_landmarks(frame, left_lm_coordinates, right_lm_coordinates, color):
    frame = frame.copy()
    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(frame, coord, 2, color, -1)
    frame = cv2.flip(frame, 1)
    return frame

def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image

def get_head_roll_angle_ears(landmarks, frame_w, frame_h):
    # Tai trái: 234, Tai phải: 454
    left_ear = denormalize_coordinates(landmarks[234].x, landmarks[234].y, frame_w, frame_h)
    right_ear = denormalize_coordinates(landmarks[454].x, landmarks[454].y, frame_w, frame_h)
    if left_ear is None or right_ear is None:
        return 0.0
    dx = right_ear[0] - left_ear[0]
    dy = right_ear[1] - left_ear[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def get_pitch_angle_nose_mouth(landmarks, frame_w, frame_h):
    nose_tip = denormalize_coordinates(landmarks[1].x, landmarks[1].y, frame_w, frame_h)
    upper_lip = denormalize_coordinates(landmarks[13].x, landmarks[13].y, frame_w, frame_h)
    lower_lip = denormalize_coordinates(landmarks[14].x, landmarks[14].y, frame_w, frame_h)
    chin = denormalize_coordinates(landmarks[152].x, landmarks[152].y, frame_w, frame_h)
    if None in (nose_tip, upper_lip, lower_lip, chin):
        return 0.0
    mouth_center = ((upper_lip[0] + lower_lip[0]) / 2, (upper_lip[1] + lower_lip[1]) / 2)
    vertical = np.linalg.norm(np.array(nose_tip) - np.array(mouth_center))
    # Bình thường đầu thẳng, vertical lớn; cúi đầu xuống, vertical nhỏ lại
    pitch_angle = (vertical - 40) * 2  # 40 là giá trị tham chiếu, có thể điều chỉnh
    return pitch_angle

def draw_important_landmarks(frame, landmarks, frame_w, frame_h, left_eye_idxs, right_eye_idxs):
    frame = frame.copy()
    # Mũi (1): đỏ
    nose_tip = denormalize_coordinates(landmarks[1].x, landmarks[1].y, frame_w, frame_h)
    if nose_tip is not None:
        cv2.circle(frame, nose_tip, 5, (0, 0, 255), -1)
    # Upper lip (13): xanh lá
    upper_lip = denormalize_coordinates(landmarks[13].x, landmarks[13].y, frame_w, frame_h)
    if upper_lip is not None:
        cv2.circle(frame, upper_lip, 5, (0, 255, 0), -1)
    # Lower lip (14): xanh dương
    lower_lip = denormalize_coordinates(landmarks[14].x, landmarks[14].y, frame_w, frame_h)
    if lower_lip is not None:
        cv2.circle(frame, lower_lip, 5, (255, 0, 0), -1)
    # Cằm (152): vàng
    chin = denormalize_coordinates(landmarks[152].x, landmarks[152].y, frame_w, frame_h)
    if chin is not None:
        cv2.circle(frame, chin, 5, (0, 255, 255), -1)
    # Các điểm quanh mắt: cam
    for idx in left_eye_idxs + right_eye_idxs:
        pt = denormalize_coordinates(landmarks[idx].x, landmarks[idx].y, frame_w, frame_h)
        if pt is not None:
            cv2.circle(frame, pt, 3, (0, 128, 255), -1)
    return frame

class VideoFrameHandler:
    def __init__(self):
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }
        self.RED = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.facemesh_model = get_mediapipe_app()
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,
            "COLOR": self.GREEN,
            "play_alarm": False,
        }
        self.EAR_txt_pos = (10, 30)
        self.head_angle = 0.0
        self.pitch_angle = 0.0
        self.drowsiness_score = 0
        # Baseline pitch
        self.baseline_pitch = None
        self.baseline_pitch_samples = []
        self.baseline_pitch_frames = 0
        self.baseline_pitch_max_frames = 60  # ~2s nếu 30fps
        # Drowsy time
        self.drowsy_time = 0.0
        self.last_drowsy = False
        self.drowsy_time_thresh = 1.5  # giây

    def process(self, frame: np.array, thresholds: dict):
        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape
        DROWSY_TIME_txt_pos = (10, int(frame_h // 2 * 1.7))
        ALM_txt_pos = (10, int(frame_h // 2 * 1.85))
        results = self.facemesh_model.process(frame)
        head_roll_angle = 0.0
        pitch_angle = 0.0
        drowsiness_score = 0
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            EAR, coordinates = calculate_avg_ear(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], frame_w, frame_h)
            self.state_tracker["last_ear"] = EAR
            # Vẽ landmark cần thiết
            frame = draw_important_landmarks(frame, landmarks, frame_w, frame_h, self.eye_idxs["left"], self.eye_idxs["right"])
            # Vẽ mesh quanh mắt như cũ
            frame = plot_eye_landmarks(frame, coordinates[0], coordinates[1], self.state_tracker["COLOR"])
            # Tính roll bằng các điểm quanh mắt
            head_roll_angle = get_head_roll_angle_ears(landmarks, frame_w, frame_h)
            self.head_angle = head_roll_angle
            # Tính pitch (dùng baseline)
            raw_pitch_angle = get_pitch_angle_nose_mouth(landmarks, frame_w, frame_h)
            # Lấy baseline pitch trong 2s đầu khi mắt mở
            if self.baseline_pitch is None and EAR > thresholds["EAR_THRESH"]:
                self.baseline_pitch_samples.append(raw_pitch_angle)
                self.baseline_pitch_frames += 1
                if self.baseline_pitch_frames >= self.baseline_pitch_max_frames:
                    self.baseline_pitch = np.mean(self.baseline_pitch_samples)
            if self.baseline_pitch is not None:
                pitch_delta = raw_pitch_angle - self.baseline_pitch
            else:
                pitch_delta = 0.0
            self.pitch_angle = pitch_delta
            # Tính drowsiness score
            if EAR < thresholds["EAR_THRESH"]:
                drowsiness_score += 1
            if abs(head_roll_angle) > 15:
                drowsiness_score += 1
            if pitch_delta < -10:
                drowsiness_score += 1
            self.drowsiness_score = drowsiness_score
            # Đếm thời gian duy trì trạng thái nguy hiểm
            now = time.perf_counter()
            if drowsiness_score >= 2:
                if self.last_drowsy:
                    self.drowsy_time += now - self.state_tracker["start_time"]
                else:
                    self.drowsy_time = 0.0
                self.last_drowsy = True
            else:
                self.drowsy_time = 0.0
                self.last_drowsy = False
            self.state_tracker["start_time"] = now
            # Cảnh báo nếu duy trì đủ lâu
            if self.drowsy_time >= self.drowsy_time_thresh:
                self.state_tracker["play_alarm"] = True
                plot_text(frame, "DROWSINESS ALERT!", ALM_txt_pos, (0, 0, 255), fntScale=1.2, thickness=3)
            else:
                self.state_tracker["play_alarm"] = False
            EAR_txt = f"EAR: {round(EAR, 2)}"
            ROLL_txt = f"Roll: {round(head_roll_angle, 1)}"
            PITCH_txt = f"Pitch: {round(pitch_delta, 1)}"
            SCORE_txt = f"Score: {drowsiness_score}"
            plot_text(frame, EAR_txt, self.EAR_txt_pos, self.state_tracker["COLOR"])
            plot_text(frame, ROLL_txt, (10, 60), (255, 0, 255))
            plot_text(frame, PITCH_txt, (10, 90), (0, 128, 255))
            plot_text(frame, SCORE_txt, (10, 120), (0, 0, 255) if drowsiness_score >= 2 else (0, 255, 0))
            DROWSY_TIME_txt = f"DROWSY: {round(self.drowsy_time, 2)}s"
            plot_text(frame, DROWSY_TIME_txt, DROWSY_TIME_txt_pos, self.state_tracker["COLOR"])
        else:
            self.state_tracker["last_ear"] = None
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["DROWSY_TIME"] = 0.0
            self.state_tracker["COLOR"] = self.GREEN
            self.state_tracker["play_alarm"] = False
            self.head_angle = 0.0
            self.pitch_angle = 0.0
            self.drowsiness_score = 0
            self.drowsy_time = 0.0
            self.last_drowsy = False
            frame = cv2.flip(frame, 1)
        return frame, self.state_tracker["play_alarm"], self.head_angle, self.pitch_angle, self.drowsiness_score

# ==== Kết thúc phần chuyển từ drowsy_detection.py ====

# Tham số ngưỡng
EAR_THRESH = 0.15
WAIT_TIME = 1.5  # giây
NO_LANDMARK_MAX_TIME = 3.0  # giây

detector = VideoFrameHandler()
thresholds = {
    "EAR_THRESH": EAR_THRESH,
    "WAIT_TIME": WAIT_TIME
}

no_landmark_time = 0.0
no_landmark_start = None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được webcam!")
    exit()

print("Nhấn Q để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được frame từ webcam!")
        break

    # Xử lý phát hiện buồn ngủ và nghiêng đầu
    result_frame, play_alarm, head_angle, pitch_angle, drowsiness_score = detector.process(frame, thresholds)

    # Kiểm tra mất landmark
    if detector.state_tracker["last_ear"] is None:
        if no_landmark_start is None:
            no_landmark_start = time.perf_counter()
        no_landmark_time = time.perf_counter() - no_landmark_start
        if no_landmark_time >= NO_LANDMARK_MAX_TIME:
            cv2.putText(result_frame, "KHONG NHAN DIEN DUOC KHUON MAT!", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)
    else:
        no_landmark_start = None
        no_landmark_time = 0.0

    # Hiển thị cảnh báo nếu phát hiện buồn ngủ hoặc nghiêng đầu
    if play_alarm:
        cv2.putText(result_frame, "BUON NGU!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    cv2.imshow("Drowsiness Detection", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 