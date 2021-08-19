import time

import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle
def gen(video_path):
    """Video streaming generator function."""
    counter = 0
    then = 0

    cap = cv2.VideoCapture(video_path)

    # Read until video is completed
    start = datetime.now()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while (cap.isOpened()):

            ret, frame = cap.read()  # import image
            if not ret:  # if vid finish repeat
                break
            if ret:  # if there is a frame continue with code

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


                image.flags.writeable = True

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

                # Rep data

                cv2.putText(image, str(counter),
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    i = then
                    # i2=then2[0]

                    new = int((landmarks[mp_pose.PoseLandmark.LEFT_HIP].y) * (640))
                    # new2=np.array((keypoints_with_scores[0][0][14][:2]*(480,640)).astype(int))[0]
                    # Calculate angle
                    angle1 = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    angle2 = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    # Curl counter logic
                    if landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > (.8):
                        if int(angle1) in range(100, 175) and int(angle2) in range(100, 175) and abs(
                                angle1 - angle2) < 10:
                            if i < new:
                                count = 1
                            if i > new:
                                if abs(new - i) > 2:
                                    if count == 1:
                                        counter += 1
                                        print(counter)
                                        count = 0

                            elif count == 0:
                                count = 0

                    # Render curl counter
                    new = int((landmarks[mp_pose.PoseLandmark.LEFT_HIP].y) * (640))
                    # new2=np.array((keypoints_with_scores[0][0][14][:2]*(480,640)).astype(int))
                    then = new
                    # then2=new2.copy()
                except:
                    pass  # Stage data


            cv2.imshow('counter', image)
            key = cv2.waitKey(1)
            if key == 27:

                break

        end = datetime.now()
        duration = (datetime.now() - start)
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver) < 3:
            fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            size = (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)

        cap.release()
        cv2.destroyAllWindows()
    return counter, duration, fps, size, start, end
