# import cv2
# cap = cv2.VideoCapture("http://ivi.bupt.edu.cn/hls/cctv1.m3u8")
# while True:
#     ret,frame = cap.read()
#     cv2.imshow("frame",frame)
#     if cv2.waitKey(41)&0xff == ord('q'):
#         break
# print('exit')
# cap.release()
# cv2.destroyAllWindows()
# import cv2
# cap = cv2.VideoCapture("http://ivi.bupt.edu.cn/hls/cctv1.m3u8")
# while True:
#     ret,frame = cap.read()
#     cv2.imshow("frame",frame)
#     if cv2.waitKey(41)&0xff == ord('q'):
#         break
# print("exit")
# cap.release()
# cv2.destroyAllWindows()
import cv2
cap = cv2.VideoCapture("http://ivi.bupt.edu.cn/hls/cctv1.m3u8")
while True:
    ret,frame = cap.read()
    cv2.imshow("CCTV",frame)
    if cv2.waitKey(40)&0xff == ord('q'):
        break
print("exit")
cap.release()
cv2.destroyAllWindows()