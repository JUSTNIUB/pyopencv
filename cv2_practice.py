import numpy as np
import cv2
import matplotlib.pyplot as plt

def GenPic2():
    img = np.empty((200,200,3),np.uint8)
    img[...,:]=(255,0,0)

    cv2.imwrite("picpic.jpg",img)
    cv2.imshow("imshow",img)
    while True:
        if cv2.waitKey(41)&0xff == ord('q'):
            break
def VideoPlay(addr):
    print(addr)
    cap = cv2.VideoCapture(addr)
    # print(cap.isOpened())
    while True:
        ret,frame = cap.read()
        # print(ret)
        cv2.imshow('frame',frame)
        if cv2.waitKey(41)&0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



#生成一张图片
#color:色值
def GenPic(color):
    img = np.empty((200,200,3),np.uint8)
    img[...,:] = color

    cv2.imwrite("picpic.jpg",img)
    cv2.imshow("picshow",img)
    cv2.waitKey(1500)
    return
#色彩空间转换
def swColorSpace():
    src = cv2.imread(r"photo/2.jpg")
    dst = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
    cv2.imshow("src show",src)
    cv2.imshow("dst show",dst)
    cv2.waitKey(0)
    return
#通道分离
def splitColor():
    img = cv2.imread(r"photo/2.jpg")
    img[...,0]=0
    img[...,1]=0
    cv2.imshow("dst show",img)
    cv2.waitKey()
    return
#颜色空间 分离特定颜色
def HsvSplit():
    img = cv2.imread("photo/1.jpg")
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_mask=np.array([120,120,180])
    upper_mask=np.array([180,255,255])

    mask = cv2.inRange(hsv_img,lower_mask,upper_mask)
    res  = cv2.bitwise_and(img,img,mask=mask)#这个啥意思?????????

    cv2.imshow('frame',img)
    cv2.imshow('mask',res)
    cv2.imshow('hsvimg', hsv_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
#获取HSV值
def getHsvValue():
    color=np.uint8([[[21,94,214]]])
    hsv_color = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
    print(hsv_color)
    return

#画各种形状
def drawShape():
    img = cv2.imread(r'photo/1.jpg')
    # cv2.line(img,(100,30),(210,80),(0,0,255),6,cv2.LINE_AA)
    cv2.circle(img,(50,50),30,(0,255,0),2,cv2.LINE_AA)
    cv2.ellipse(img,(100,100),(100,50),90,0,360,(255,255,0),2,cv2.LINE_AA)
    cv2.rectangle(img,(100,30),(200,90),(255,0,0),2,cv2.LINE_AA)
    cv2.imshow("Shape like",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def drawShape2():
    img = cv2.imread(r'photo/1.jpg')
    pts = np.array([[10,5],[50,10],[70,20],[20,30]],np.int32)

    # pts = pts.reshape((-1,1,2))
    # cv2.polylines(img,[pts],True,(0,0,255),3)
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    cv2.putText(img,'beautiful girl',(20,30),font,1,(0,0,255),1,cv2.LINE_AA)

    cv2.imshow("hello",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def OTSU_Test():
    img = cv2.imread("photo/1.jpg")
    print(img.shape)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    cv2.imshow("origin pic",img)
    cv2.imshow("gray",gray)
    cv2.imshow("bianry",binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def OTSU_ShowAllEffct():
    #造一张400*400*3的图片
    imgdata = np.empty((256,256,3),dtype=np.float)#hwc????
    for i in range(256):
        imgdata[:,i]= 256-i
    print(imgdata)
    cv2.imwrite('photo/6.jpg',imgdata)
    # imgdata = cv2.imread('photo/6.jpg')
    # cv2.imshow('otsu photo',imgdata)
    # cv2.waitKey(0)


    img = cv2.imread('photo/6.jpg',0)
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

    titles = ['Original Image','Binary','Binary_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]

    for i in range(6):
        plt.subplot(2,3,i+1)
        print(i)
        plt.imshow(images[i],'gray')#gray代表灰度图?
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

def OTSU_AdaptiveTher():
    img = cv2.imread('photo/7.jpg',0)
    img = cv2.GaussianBlur(img,(5,5),0)#这个是设置高斯核，不清楚有啥用

    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    #均匀# 用于处理不同区域亮度不一致时的图片
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,81,2)
    #高斯
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,81,2)

    titles = ['Original Image','Global Thresholding(v=127)','Adaptive Mean Thresholding','Adaptive Gaussian Thresholding']
    images = [img,th1,th2,th3]
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()
def u8AddSub():
    x = np.uint8([250])
    y = np.uint8([10])
    print(cv2.add(x,y))
    print(cv2.subtract(x,y))



#图像混合:第二张图片是黑白的
def PicMix1():
    img = cv2.imread("photo/cat.jpg")
    img2 = cv2.imread("photo/8.jpg")
    w,h,_= img.shape
    img2 = cv2.resize(img2,(h,w))

    img = cv2.addWeighted(img,0.7,img2,0.5,0)
    print(img.shape)

    cv2.imshow("test",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#保留图片 原有的色彩
def PicMix2():
    img1 = cv2.imread('photo/cat.jpg')
    img2 = cv2.imread('photo/9.jpg')
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)  # 这个是设置高斯核，不清楚有啥用
    rows,cols,channels = img2.shape
    img1 = cv2.resize(img1,(cols,rows))

    roi = img1[0:rows,0:cols]
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    # mask = cv2.adaptiveThreshold(img2gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    ret,mask = cv2.threshold(img2gray,200,255,cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # cv2.imshow("mask_inv", mask_inv)
    cv2.waitKey(0)
    img1_bg = cv2.bitwise_and(roi,roi,mask=mask)#前景

    img2_bg = cv2.bitwise_and(img2,img2,mask=mask_inv)#后景

    dst = cv2.add(img1_bg,img2_bg)
    img1[0:rows,0:cols] = dst
    cv2.imshow("mask_inv",img1)
    cv2.waitKey(0)
#放射变换
def AffineChange():
    src=cv2.imread('photo/gakki.jpg')


    #src=cv2.resize(src,(200,300))
    rows, cols, channel = src.shape
    print(src.shape)
    # M = np.float32([[1,0,50],[0,1,50]]) #平移
    # M = np.float32([[0.5,0,0],[0,0.5,0]]) #缩小
    #M = np.float32([[-0.5,0,rows//2],[0,0.5,0]])
    M = np.float32([[1,0.5,0],[0,1,0]])
    # M = cv2.getRotationMatrix2D((cols/2,rows/2),45,0.7)

    #行列相反
    dst = cv2.warpAffine(src,M,(1000,1000))#最后一个参数输出图像的大小,不是缩放，而是像素点的多少

    cv2.imshow('src pic',src)
    cv2.imshow('dst pic',dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


#放大某个区域
def PerspectiveChange():
    img = cv2.imread("photo/gakki.jpg")
    pts1 = np.float32([[180,0],[400,0],[180,400],[400,400]])
    pts2 = np.float32([[0,0],[440,0],[0,800],[440,800]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(500,800))

    cv2.imshow("src",img)
    cv2.imshow("dst",dst)
    cv2.waitKey(0)

#膨胀 放大：在操作之前要二值化
def DilateOp():
    img = cv2.imread("photo/8.jpg",cv2.IMREAD_GRAYSCALE)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dst = cv2.dilate(img,kernel)
    cv2.imshow('Dilate',dst)
    cv2.imshow('Original',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#腐蚀
def ErodeOp():
    img = cv2.imread("photo/8.jpg",cv2.IMREAD_GRAYSCALE)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dst = cv2.erode(img,kernel)
    cv2.imshow('Erode',dst)
    cv2.imshow('Original',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#开操作
def Openop():
    img=cv2.imread("photo/10.jpg",0)
    ret, img = cv2.threshold(img, 200, 255, cv2.THRESH_TRUNC)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dst = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel,iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst2 = cv2.morphologyEx(dst,cv2.MORPH_CLOSE,kernel,iterations=1)


    cv2.imshow('src',img)
    cv2.imshow('dst',dst)
    cv2.imshow("dst2",dst2)
    cv2.waitKey(0)
def Closeop():
    img=cv2.imread("photo/10.jpg",0)
    ret,img=cv2.threshold(img,200,255,cv2.THRESH_TRUNC)
    # img=cv2.GaussianBlur(img,(5,5),0)
    # img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,111,2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dst = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel,iterations=1)

    cv2.imshow('src',img)
    cv2.imshow('dst',dst)
    cv2.waitKey(0)



# ErodeOp()
# DilateOp()
# Closeop()
Openop()
# PerspectiveChange()
# AffineChange()



# PicMix2()
# PicMix()
# u8AddSub()
# OTSU_AdaptiveTher()
# OTSU_ShowAllEffct()
# OTSU_Test()
#"http://ivi.bupt.edu.cn/hls/cctv1.m3u8"  "d:/6.source/0.movie/mp4/love death and robot 01.mp4" "d:/6.source/0.movie/mp4/01.mp4"
# VideoPlay(r"D:/6.source/0.movie/mp4/01.mp4")
# drawShape2()
# drawShape()
# GenPic2()
# HsvSplit()
# getHsvValue()
# GenPic((80,255,120))
# swColorSpace()
# splitColor()