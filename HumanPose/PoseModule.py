import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self,mode = False,
                 upBody = False,
                 smooth = True,
                 detectionCon = 0.5,
                 trackCon = 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody,
                                     self.smooth,self.detectionCon,self.trackCon)

    def findPose(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #conversion
        self.results = self.pose.process(imgRGB)

    def findPosition(self, img):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList

def main():
    cap = cv2.VideoCapture('PoseVedio/test.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img,draw=False)
        print(lmList[16])
        cv2.circle(img, (lmList[16][1], lmList[16][2]), 5, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (lmList[15][1], lmList[15][2]), 5, (255, 0, 0), cv2.FILLED)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()