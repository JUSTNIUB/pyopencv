import cv2
from PIL import Image

img = cv2.imread("photo/2.jpg",cv2.IMREAD_GRAYSCALE)
# img = img[...,::-1]
# img = Image.fromarray(img)
# img.show()
cv2.imshow("pic show",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

