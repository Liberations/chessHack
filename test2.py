import cv2 as cv
import numpy as np
import glob
import os
from PIL import ImageFont, ImageDraw, Image
import math

TYPE_RED = True #定义是否红方

# 定义要裁剪的区域坐标
CHESS_LEFT = 0  # 左上角 x 坐标
CHESS_TOP = 0  # 左上角 y 坐标
CHESS_WIDTH = 1080 #棋盘的宽度 用来计算每个棋子的位置
CHESS_HEIGHT = 2400 #棋盘的高度
CHESS_MAX_RADIUS = 30 #每个棋子的最大半径



class Chess():
    name = "未知"
    x = 0 #横坐标转成0~8
    y = 0 #纵坐标转成0~9
    reallyX = 0
    reallyY = 0
    isDied = False #是否已经下场
    def __init__(self,name,x,y):
        self.name = name
        self.x = x
        self.y = y
    def __repr__(self):
        return f"({self.name}, {self.reallyX}, {self.reallyY})"

# 上方 所有棋子
baseTopChessList = []
baseTopChessList.append(Chess("黑车", 0, 0))
baseTopChessList.append(Chess("黑马", 0, 1))
baseTopChessList.append(Chess("黑相", 0, 2))
baseTopChessList.append(Chess("黑士", 0, 3))
baseTopChessList.append(Chess("黑帅", 0, 4))
baseTopChessList.append(Chess("黑士", 0, 5))
baseTopChessList.append(Chess("黑相", 0, 6))
baseTopChessList.append(Chess("黑马", 0, 7))
baseTopChessList.append(Chess("黑车", 0, 8))
baseTopChessList.append(Chess("黑炮", 2, 1))
baseTopChessList.append(Chess("黑炮", 2, 7))
baseTopChessList.append(Chess("黑兵", 3, 0))
baseTopChessList.append(Chess("黑兵", 3, 2))
baseTopChessList.append(Chess("黑兵", 3, 4))
baseTopChessList.append(Chess("黑兵", 3, 6))
baseTopChessList.append(Chess("黑兵", 3, 8))

#下方棋子
baseBottomChessList = []
baseBottomChessList.append(Chess("红车", 9, 0))
baseBottomChessList.append(Chess("红马", 9, 1))
baseBottomChessList.append(Chess("红相", 9, 2))
baseBottomChessList.append(Chess("红士", 9, 3))
baseBottomChessList.append(Chess("红帅", 9, 4))
baseBottomChessList.append(Chess("红士", 9, 5))
baseBottomChessList.append(Chess("红相", 9, 6))
baseBottomChessList.append(Chess("红马", 9, 7))
baseBottomChessList.append(Chess("红车", 9, 8))
baseBottomChessList.append(Chess("红炮", 7, 1))
baseBottomChessList.append(Chess("红炮", 7, 7))
baseBottomChessList.append(Chess("红兵", 6, 0))
baseBottomChessList.append(Chess("红兵", 6, 2))
baseBottomChessList.append(Chess("红兵", 6, 4))
baseBottomChessList.append(Chess("红兵", 6, 6))
baseBottomChessList.append(Chess("红兵", 6, 8))

redChessList = []
blacChessList = []

lastRedChessList = []
lastBlacChessList = []

def resizeToPosX(x):
    chessSize = 8*x/CHESS_WIDTH
    return math.floor(chessSize)

def resizeToPosY(y):
    chessSize = 10*y/CHESS_HEIGHT
    return math.floor(chessSize)

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
        result = compare_images(image, target_image, threshold=0.9)
        if result > res:
            res = result
            index = i
    
    return index  # 如果未找到匹配的图像，返回-1


def compare_images(image1, image2, threshold=0.9):
  # 转换为灰度图像
    #gray_big = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    #gray_small = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    # 使用模板匹配算法
    results = cv.matchTemplate(image2, image1, cv.TM_CCOEFF_NORMED)
    # 获取匹配结果（最大值和对应坐标）
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(results)
    # 设置阈值判断是否匹配成功
    if max_val > threshold:
        return max_val
    else:
        return -1

#保存分割后的象棋
def saveChessCutImg(chessList,x,y,r,circle_image):
      for index,chess in enumerate(chessList):
          if chess.x==resizeToPosX(x) and chess.y==resizeToPosY(y):
            name = '{}/{}_{}_{}.jpg'.format(CHESS_CUT_PATH,index,chess.x,chess.y)
            print("将要保存的路径",name)
              # 定义要裁剪的区域坐标
            x = 0  # 左上角 x 坐标
            y = 0  # 左上角 y 坐标
            width = r*2  # 裁剪区域宽度
            height = r*2-10  # 裁剪区域高度
            # 裁剪图像
            img = circle_image[y:y+height, x:x+width]
            cv.imwrite(name,img)

def recognizeRed(x,y,r,image):
    # 圆形区域的中心坐标和半径
    center = (x, y)
    radius = r

    # 获取圆形区域的图像副本
    circle_image = image[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius].copy()


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
    position = [resizeToPosX(x), resizeToPosY(y)]
    #print("找到棋子reallyX {} reallyY {} resizeToPosX {} resizeToPosY {}".format(x,y,position[0],position[1]))
    if red_pixel_ratio > red_ratio_threshold:
        redChess = Chess("红棋子",position[0],position[1])
        redChess.reallyX = x
        redChess.reallyY = y
        redChessList.append(redChess)
    else:
        blackChess = Chess("黑棋子",position[0],position[1])
        blackChess.reallyX = x
        blackChess.reallyY = y
        blacChessList.append(blackChess)
        

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
    print("每个棋子的半径",CHESS_MAX_RADIUS)
    # 需要调整的参数：dp、minDist、param1、param2、minRadius、maxRadius
    dp = 1  # 累加器分辨率与图像分辨率的倒数之比
    minDist = CHESS_MAX_RADIUS-10  # 检测到的圆心之间的最小距离
    param1 = 200  # Canny 边缘检测的高阈值
    param2 = 80  # 累加器阈值，较小的值会导致更多的假阳性圆形
    minRadius = math.floor(CHESS_MAX_RADIUS/2)  # 圆的最小半径
    maxRadius = math.floor(CHESS_MAX_RADIUS*0.8)  # 圆的最大半径

    # 在灰度图像中检测圆形
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    # 确保至少检测到一个圆形
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)

        lastBlacChessList.clear()
        lastBlacChessList.extend(blacChessList)
        lastRedChessList.clear()
        lastRedChessList.extend(redChessList)

        redChessList.clear()
        blacChessList.clear()
    
        for (x, y, r) in circles:
                isRed = recognizeRed(x,y,r,image)
                if isRed:
                    cv.circle(image, (x, y), r, (0, 255, 0), 2)
                else:
                    cv.circle(image, (x, y), r, (255, 0, 0), 2)

        lastBlacChessList1 = sorted(lastBlacChessList, key=lambda c: (c.reallyX, c.reallyY))
        lastRedChessList1 = sorted(lastRedChessList, key=lambda c: (c.reallyX, c.reallyY))

        redChessList1 = sorted(redChessList, key=lambda c: (c.reallyX, c.reallyY))
        blacChessList1 = sorted(blacChessList, key=lambda c: (c.reallyX, c.reallyY))

        print("上次黑色棋子",lastBlacChessList1) 
        print("上次红色棋子",lastRedChessList1)  
         
            
        print("总共找到红色棋子",redChessList1)  
        print("总共找到黑色棋子",blacChessList1)    

    return image

#开始
img = cv.imread('chessboard2.jpg')
height, width, channels = img.shape
# width/height = 9/10
CHESS_WIDTH = width
CHESS_MAX_RADIUS = math.floor(width/9/2)
CHESS_HEIGHT = math.floor(width*10/9)
CHESS_TOP = math.ceil((height-CHESS_HEIGHT)/2)
CHESS_CUT_PATH = "cut_{}".format(CHESS_WIDTH)
localImags = ['chessboard1.jpg','chessboard2.jpg','chessboard3.jpg']
if __name__ == '__main__':
    for path in localImags:
        img = cv.imread(path)
        height, width, channels = img.shape
        # width/height = 9/10
        CHESS_WIDTH = width
        CHESS_MAX_RADIUS = math.floor(width/9/2)
        CHESS_HEIGHT = math.floor(width*10/9)
        CHESS_TOP = math.ceil((height-CHESS_HEIGHT)/2)
        CHESS_CUT_PATH = "cut_{}".format(CHESS_WIDTH)
        # 裁剪图像
        img = img[CHESS_TOP:CHESS_TOP+CHESS_HEIGHT, CHESS_LEFT:CHESS_LEFT+CHESS_WIDTH]
        #img = np.array(img)
        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


        # 识别圆的位置S
        img = recognize(img)
        for chess in redChessList:
            #print(chess.name)
            img = cv2ImgAddText(img,chess.name, chess.reallyX,chess.reallyY,textColor=(255, 0, 0), textSize=40)

        for chess in blacChessList:
            #print(chess.name)
            img = cv2ImgAddText(img,chess.name, chess.reallyX,chess.reallyY,textColor=(0, 0, 0),  textSize=40)
        
        # 调整图像尺寸
        resized_image = cv.resize(img, (math.floor(CHESS_WIDTH/2), math.floor(CHESS_HEIGHT/2)))  # 设置目标宽度和高度
        # 显示结果图像
        #cv.imshow('Chessboard', resized_image)
        #cv.waitKey(0)
        #cv.destroyAllWindows()