import cv2
from PIL import Image

src = cv2.imread(r"photo/2.jpg")
dst = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
cv2.imshow("src show",src)
cv2.imshow("dst show",dst)
cv2.waitKey(0)