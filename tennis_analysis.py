import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np


# ===============================
# body axis
# ===============================
def draw_body_axis(frame, landmarks):

    h, w, _ = frame.shape

    def get_point(idx):
        lm = landmarks.landmark[idx]
        return int(lm.x * w), int(lm.y * h)

    ls = get_point(11)
    rs = get_point(12)
    lh = get_point(23)
    rh = get_point(24)

    shoulder_mid = ((ls[0] + rs[0]) // 2,
                    (ls[1] + rs[1]) // 2)

    hip_mid = ((lh[0] + rh[0]) // 2,
               (lh[1] + rh[1]) // 2)

    cv2.line(frame, shoulder_mid, hip_mid, (0, 0, 255), 2)

    ref_top = (hip_mid[0], hip_mid[1] - 150)

    cv2.line(frame, hip_mid, ref_top, (255, 255, 255), 2)

    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]

    angle = np.degrees(np.arctan2(dx, -dy))

    cv2.putText(
        frame,
        f"{angle:.1f} deg",
        (shoulder_mid[0] + 10, shoulder_mid[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # shoulder angle
    dx_s = ls[0] - rs[0]
    dy_s = ls[1] - rs[1]

    shoulder_angle = np.degrees(
        np.arctan2(-dy_s, dx_s)
    )

    cv2.putText(
        frame,
        f"{shoulder_angle:.1f}",
        (rs[0] + 30, rs[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


# ===============================
# center of mass
# ===============================
def draw_center_of_mass(frame, landmarks):

    h, w, _ = frame.shape

    def get_point(idx):
        lm = landmarks.landmark[idx]
        return np.array([lm.x * w, lm.y * h])

    ls = get_point(11)
    rs = get_point(12)
    lh = get_point(23)
    rh = get_point(24)

    shoulder_mid = (ls + rs) / 2
    hip_mid = (lh + rh) / 2

    com = 0.6 * hip_mid + 0.4 * shoulder_mid

    cx, cy = int(com[0]), int(com[1])

    cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)

    return (cx, cy)


# ===============================
# COM axis
# ===============================
def draw_com_axis(frame, landmarks, com):

    h, w, _ = frame.shape

    def get_point(idx):
        lm = landmarks.landmark[idx]
        return np.array([lm.x * w, lm.y * h])

    la = get_point(27)
    ra = get_point(28)

    foot_mid = (la + ra) / 2

    fx, fy = int(foot_mid[0]), int(foot_mid[1])

    cx, cy = com

    cv2.line(frame, (cx, cy), (cx, fy), (0, 255, 255), 2)

    cv2.line(frame, (fx - 50, fy), (fx + 50, fy), (255, 255, 255), 2)

    if cx > fx + 10:
        status = "BACK"
        color = (255, 255, 0)

    elif cx < fx - 10:
        status = "FORE"
        color = (0, 0, 255)

    else:
        status = "CENTER"
        color = (255, 255, 255)

    cv2.putText(
        frame,
        status,
        (cx + 10, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


# ===============================
# main process
# ===============================
def process_video(video_path, frame_placeholder):

    model = YOLO("best.pt")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(model_complexity=2)

    cap = cv2.VideoCapture(video_path)

    ball_positions = []

    prev_ball = None
    speed_threshold = 6

    prev_gray = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        motion_mask = None

        if prev_gray is not None:

            diff = cv2.absdiff(gray, prev_gray)

            _, motion_mask = cv2.threshold(
                diff,
                25,
                255,
                cv2.THRESH_BINARY
            )

        prev_gray = gray

        # =====================
        # YOLO
        # =====================

        results = model(frame, verbose=False)[0]

        ball_center = None
        racket_center = None

        for box in results.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf > 0.4:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # tennis ball
                if cls == 0:

                    candidate = (cx, cy)

                    moving = False

                    if motion_mask is not None:
                        if motion_mask[cy, cx] > 0:
                            moving = True

                    if prev_ball is None:

                        ball_center = candidate
                        prev_ball = candidate

                    else:

                        dx = candidate[0] - prev_ball[0]
                        dy = candidate[1] - prev_ball[1]

                        speed = np.sqrt(dx * dx + dy * dy)

                        if speed > speed_threshold and moving:

                            ball_center = candidate
                            prev_ball = candidate

                    if ball_center is not None:

                        cv2.circle(
                            frame,
                            ball_center,
                            5,
                            (0, 255, 0),
                            -1
                        )

                # racket
                elif cls == 1:

                    racket_center = (cx, cy)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 255),
                        2
                    )

                    cv2.circle(
                        frame,
                        racket_center,
                        5,
                        (255, 0, 255),
                        -1
                    )

        # =====================
        # trajectory
        # =====================

        if ball_center is not None:
            ball_positions.append(ball_center)

        for i in range(1, len(ball_positions)):

            cv2.line(
                frame,
                ball_positions[i - 1],
                ball_positions[i],
                (255, 0, 0),
                1
            )

        # =====================
        # pose estimation
        # =====================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        pose_result = pose.process(rgb)

        if pose_result.pose_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                pose_result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            draw_body_axis(
                frame,
                pose_result.pose_landmarks
            )

            com = draw_center_of_mass(
                frame,
                pose_result.pose_landmarks
            )

            draw_com_axis(
                frame,
                pose_result.pose_landmarks,
                com
            )

        # =====================
        # streamlit display
        # =====================

        frame_placeholder.image(
            frame,
            channels="BGR"
        )

    cap.release()