import cv2 as cv
import numpy as np
import glob
import os
from PIL import ImageFont, ImageDraw, Image
import math
import operator

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
        return f"({self.name}, {self.reallyX}, {self.reallyY}, {self.x}, {self.y})"

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

chessList = [] #本次识别到的棋子
lastChessList = []

chessMatri =  [[0] * 10 for _ in range(9)] #本次识别到的棋子矩阵
lastChessMatri =  [[0] * 10 for _ in range(9)] #本次识别到的棋子矩阵

def resizeToPosX(x):
    #(2*x+1)*radius = y;
    
    chessSize = (x/CHESS_MAX_RADIUS-1)/2
    return math.floor(chessSize)

def resizeToPosY(y):
    chessSize = (y/CHESS_MAX_RADIUS-1)/2
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

      
def create_matrix(chess_list):
    matrix = [["空" for _ in range(9)] for _ in range(8)]  # 创建一个空的 8x9 矩阵

    for chess in chess_list:
        x = chess.x
        y = chess.y
        matrix[x][y] = chess.name  # 根据 Chess 对象的 x 和 y 值在矩阵中设置标志为 1

    return matrix

def print_matrix(matrix):
    for row in matrix:
        row_str = "\t".join(str(element) for element in row)
        print(row_str)

def recognize(image):
    # 读取图像
    # 将图像转换为灰度
    #gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    print("每个棋子的半径",CHESS_MAX_RADIUS)
    # 需要调整的参数：dp、minDist、param1、param2、minRadius、maxRadius
    dp = 1  # 累加器分辨率与图像分辨率的倒数之比
    minDist = CHESS_MAX_RADIUS-10  # 检测到的圆心之间的最小距离
    param1 = 200  # Canny 边缘检测的高阈值
    param2 = 80  # 累加器阈值，较小的值会导致更多的假阳性圆形
    minRadius = math.floor(CHESS_MAX_RADIUS/2)  # 圆的最小半径
    maxRadius = math.floor(CHESS_MAX_RADIUS*0.8)  # 圆的最大半径

    # 在灰度图像中检测圆形
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    # 确保至少检测到一个圆形
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)

        lastChessList.clear()
        lastChessList.extend(chessList)
        
        
        lastChessMatri.clear()
       
    
        for (x, y, r) in circles:
                 position = [resizeToPosX(x), resizeToPosY(y)]
                 redChess = Chess("红",position[0],position[1])
                 redChess.reallyX = x
                 redChess.reallyY = y
                 chessList.append(redChess)
                 cv.circle(image, (x, y), r, (255, 255, 0), 5)

        chessMatri = create_matrix(chessList)

    

        #print_matrix(lastChessMatri)
        print_matrix(chessMatri)    

    return image

#开始
img = cv.imread('chessboard2.jpg')
height, width, channels = img.shape
# width/height = 9/10
CHESS_WIDTH = width
CHESS_MAX_RADIUS = 0
CHESS_HEIGHT = 0
CHESS_TOP = 0
CHESS_CUT_PATH = ""
localImags = ['chessboard2.jpg']
if __name__ == '__main__':
    for path in localImags:
        img = cv.imread(path)
        # 进行边界检测
        img = cv.Canny(img, 50, 150)
        height, width = img.shape
        # width/height = 9/10
        CHESS_WIDTH = width
        CHESS_MAX_RADIUS = math.floor(width/9/2)
        CHESS_HEIGHT = math.floor(CHESS_MAX_RADIUS*10*2)
        CHESS_TOP = math.ceil((height-CHESS_HEIGHT)/2)
        CHESS_CUT_PATH = "cut_{}".format(CHESS_WIDTH)
        # 裁剪图像
        img = img[CHESS_TOP:CHESS_TOP+CHESS_HEIGHT, CHESS_LEFT:CHESS_LEFT+CHESS_WIDTH]
        #img = np.array(img)
        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        

        # 识别圆的位置S
        img = recognize(img)
        for chess in chessList:
            #print(chess.name)
            img = cv2ImgAddText(img,chess.name, chess.reallyX,chess.reallyY,textColor=(0, 0, 255), textSize=40)

    
        chessCenterX = math.floor(CHESS_WIDTH/2)
        chessCenterY = math.floor(CHESS_HEIGHT/2)
        cv.circle(img, (chessCenterX, chessCenterY), math.floor(CHESS_MAX_RADIUS), (255, 0, 0), 15)
        #-8 -6 -4 -2 0 2 4 6 8 
        #-10 -8 -6 -4 -2 0 2 4 6 8 10
        chessRadius = CHESS_MAX_RADIUS - 15
        for cloumn in range (0,10):
             for row in range(0,9):
                posX = math.floor((row-4)*2*CHESS_MAX_RADIUS+chessCenterX)
                posY =  math.floor((cloumn-5)*2*CHESS_MAX_RADIUS+chessCenterY+CHESS_MAX_RADIUS)
                print("位置{} {}".format(posX,posY))
                cv.circle(img, (posX, posY),chessRadius, (255, 255, 0), 2)
        
        # 调整图像尺寸
        resized_image = cv.resize(img, (math.floor(CHESS_WIDTH/2), math.floor(CHESS_HEIGHT/2)))  # 设置目标宽度和高度
        # 显示结果图像
        cv.imshow('Chessboard', resized_image)
        cv.waitKey(0)
        cv.destroyAllWindows()