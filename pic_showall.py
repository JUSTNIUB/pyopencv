import os
from PIL import Image
import matplotlib.pyplot as plt


for picpath in os.listdir("photo"):
    plt.cla()
    img = Image.open(os.path.join("photo/",picpath))
    plt.imshow(img)
    plt.show(block=False)
    plt.pause(1)
