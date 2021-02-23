import cv2
cap = cv2.VideoCapture('./hr/data/data0827/sub1/0_1/face/check.avi')
cap.isOpened()
while(True):
    ret, frame = cap.read()
    print(ret)
    if ret:
        print(frame)
        cv2.imshow('test', frame)
        cv2.waitKey(1)
        cap.release()