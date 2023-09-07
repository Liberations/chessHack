import cv2 as cv
import numpy as np
import glob
import os
from PIL import ImageFont, ImageDraw, Image
import math

TYPE_RED = True #定义是否红方

# 定义要裁剪的区域坐标
CHESS_LEFT = 0  # 左上角 x 坐标
CHESS_TOP = 600  # 左上角 y 坐标
CHESS_WIDTH = 1080 #棋盘的宽度 用来计算每个棋子的位置
CHESS_HEIGHT = 1200 #棋盘的高度

class Chess():
    name = "未知"
    x = 0 #横坐标转成0~8
    y = 0 #纵坐标转成0~9
    reallyX = 0
    reallyY = 0
    isDied = False #是否已经下场
    def __init__(self,name,x,y):
        self.name = name
        self.type = type

redChessList = []
blacChessList = []

def resizeToPosX(x):
    chessSize = CHESS_WIDTH/9
    x = x/chessSize
    return math.floor(x)

def resizeToPosY(y):
    chessSize = CHESS_HEIGHT/9
    y = y/chessSize
    return math.floor(y)

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)


def read_files(folder_path):
    image_files = glob.glob(folder_path + '/*.jpg')
    return image_files

def read_images(folder_path):
    image_files = glob.glob(folder_path + '/*.jpg')
    images = []
    
    for file in image_files:
        #print("正在读取文件:", file)
        #image = cv.imread(file)
        image = cv.imdecode(np.fromfile(file, dtype=np.uint8), -1)
        images.append(image)
    
    return images

def find_matching_image(target_image, image_list):
    res = -1
    index = -1
    for i, image in enumerate(image_list):
        # 进行图像匹配操作，此处使用模板匹配的方法
        # ...

        # 假设匹配结果为 result
        result = compare_images(image, target_image, threshold=0.8)
        if result >= res:
            res = result
            index = i
    
    return index  # 如果未找到匹配的图像，返回-1

def cutImages():
    for i, image in enumerate(images):
         # 定义要裁剪的区域坐标
        x = 10  # 左上角 x 坐标
        y = 10  # 左上角 y 坐标
        width = 50  # 裁剪区域宽度
        height = 60  # 裁剪区域高度

        # 裁剪图像
        img = image[y:y+height, x:x+width]
        file_name = os.path.basename(files[i])
        name = 'chesscut/a_{}.jpg'.format(i)
        cv.imwrite(name,img)


# 读取多个图片并保存在数组中
image_folder = 'chesscut'
files = read_files(image_folder)
images = read_images(image_folder)
#cutImages()
#print("初始化棋子{}".format(files))

def compare_images(image1, image2, threshold=0.9):
  # 转换为灰度图像
    #gray_big = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    #gray_small = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    # 使用模板匹配算法
    results = cv.matchTemplate(image2, image1, cv.TM_CCOEFF_NORMED)
    # 获取匹配结果（最大值和对应坐标）
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(results)
    # 设置阈值判断是否匹配成功
    return max_val

def recognizeRed(x,y,r,image):
    # 圆形区域的中心坐标和半径
    center = (x, y)
    radius = r

    # 获取圆形区域的图像副本
    circle_image = image[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius].copy()
    # 保存图像
    #name = 'chesscut/{}_{}_{}.jpg'.format(x,y,r)
    #cv.imwrite(name, circle_image)
    # 查找匹配的图像
    matching_index = find_matching_image(circle_image, images)

   
    # 将图像副本转换为HSV颜色空间
    hsv_circle = cv.cvtColor(circle_image, cv.COLOR_BGR2HSV)

    # 定义红色的色调范围
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # 使用颜色阈值分割提取红色区域
    red_mask1 = cv.inRange(hsv_circle, lower_red1, upper_red1)
    red_mask2 = cv.inRange(hsv_circle, lower_red2, upper_red2)
    red_mask = cv.bitwise_or(red_mask1, red_mask2)

    # 计算红色区域的像素比例
    red_pixel_ratio = np.count_nonzero(red_mask) / (red_mask.shape[0] * red_mask.shape[1])

    # 设置红色像素比例阈值
    red_ratio_threshold = 0.2
    if matching_index != -1:
        file_name = os.path.basename(files[matching_index]).replace('.jpg', '')
        #print("匹配到图片", file_name)
        text = "{}".format(file_name)
        position = [resizeToPosX(x), resizeToPosX(y)]
        #print("{}位置{}".format(file_name,position))
        if red_pixel_ratio > red_ratio_threshold:
            redChess = Chess(file_name,position[0],position[1])
            redChess.reallyX = x
            redChess.reallyY = y
            redChessList.append(redChess)
            cv2ImgAddText(image,text, x,y,(0, 0, 255), 20)
        else:
            blackChess = Chess(file_name,position[0],position[1])
            blackChess.reallyX = x
            blackChess.reallyY = y
            blacChessList.append(blackChess)
            cv2ImgAddText(image,text, x,y,(0, 0, 0), 20)

    # 判断圆形区域是否包含红色
    if red_pixel_ratio > red_ratio_threshold:
       # print("圆形区域包含红色")
        return True
    else:
       # print("圆形区域不包含红色")
        return False
        

def recognize(image):
    # 读取图像
    # 将图像转换为灰度
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 需要调整的参数：dp、minDist、param1、param2、minRadius、maxRadius
    dp = 1  # 累加器分辨率与图像分辨率的倒数之比
    minDist = 30  # 检测到的圆心之间的最小距离
    param1 = 200  # Canny 边缘检测的高阈值
    param2 = 80  # 累加器阈值，较小的值会导致更多的假阳性圆形
    minRadius = 10  # 圆的最小半径
    maxRadius = 50  # 圆的最大半径

    # 在灰度图像中检测圆形
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    # 确保至少检测到一个圆形
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)

        redChessList.clear()
        blacChessList.clear()
     # 绘制检测到的圆形
        for (x, y, r) in circles:
            isRed = recognizeRed(x,y,r,image)
            if isRed:
                cv.circle(image, (x, y), r, (0, 0, 255), 2)
            else:
                cv.circle(image, (x, y), r, (255, 0, 0), 2)

    return image
img = cv.imread('chessboard.jpg')
if __name__ == '__main__':
    # 裁剪图像
    img = img[CHESS_TOP:CHESS_TOP+CHESS_HEIGHT, CHESS_LEFT:CHESS_LEFT+CHESS_WIDTH]
    #img = np.array(img)
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # 识别圆的位置
    img = recognize(img)
    for chess in redChessList:
        print(chess.name)
        img = cv2ImgAddText(img,chess.name, chess.reallyX+15,chess.reallyY,textColor=(255, 0, 0), textSize=40)

    for chess in blacChessList:
        print(chess.name)
        img = cv2ImgAddText(img,chess.name, chess.reallyX+15,chess.reallyY,textColor=(0, 0, 0),  textSize=40)
    
    # 调整图像尺寸
    resized_image = cv.resize(img, (540, 600))  # 设置目标宽度和高度
    # 显示结果图像
    cv.imshow('Chessboard', resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()