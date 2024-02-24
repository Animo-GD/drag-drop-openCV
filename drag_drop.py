import cv2
import numpy as np
from hand_tracking_module import hand_detector


class drag_rect:
    def __init__(self, center: tuple, size=(200, 200)):
        self.center = center
        self.size = size

    def update(self, cursor: tuple,dist:float):
        cx, cy = self.center
        w, h = self.size
        # If the index of the fingertip is in the rectangle region

        if (cx - w // 2 < cursor[0] < cx + w // 2 and
            cy - h // 2 < cursor[1] < cy + h // 2) and dist < 50:
            self.center = cursor
            return True
        return False


cap = cv2.VideoCapture(0)
cap.set(1, 40)
cap.set(3, 1280)
cap.set(4, 720)
detector = hand_detector(min_detection_confidence=.75)
# Rectangle Dimensions
rect_color = (255, 0, 255)

# Creating Four Rectangles
rect_list = []
for x in range(5):
    rect_list.append(drag_rect((x*250+150,150)))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = detector.find_hand(frame)
    landmarks = detector.find_position(frame)
    if landmarks:
        finger1 = (landmarks[8][1], landmarks[8][2])
        finger2 = (landmarks[4][1], landmarks[4][2])
        # Getting The Distance Between the two fingers
        dist = np.hypot(finger2[0] - finger1[0], finger2[1] - finger1[1])
        # Updating The Finger Position Relating To The Rectangle
        catch = False
        for rect in rect_list:
            catch = rect.update(finger1,dist)
    # Drawing The Rectangle
    img_new = np.zeros_like(frame,np.uint8)
    for rect in rect_list:
        cx,cy = rect.center
        w,h = rect.size
        cv2.rectangle(img_new, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), rect_color, -1)

    out = frame.copy()
    alpha = 0.6
    mask = img_new.astype(bool)
    out[mask] = cv2.addWeighted(frame,alpha,img_new,1-alpha,0)[mask]
    cv2.imshow("cam", out)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
