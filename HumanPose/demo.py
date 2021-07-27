import cv2
import time
import PoseModule as pm

def main():
    cap = cv2.VideoCapture('PoseVedio/test.mp4')
    pTime = 0
    detector = pm.poseDetector()
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