import cv2
import numpy as np
import time
import os
import requests
import sys

# 获取传递给脚本的参数
device_ip = sys.argv[1]
print("本脚本的IP",device_ip)
if device_ip is None:
    exit
#本机是否红方
isRed = None


#通过矩阵的x,y计算实际点击的位置
def tapPos(row,col):
    tapX = getPosX(col)
    tapY = getPosY(row)
    os.system("adb -s {} shell input tap {} {}".format(device_ip,tapX,tapY))
    print("点击位置{},{} {},{}".format(row,col,tapX,tapY))
    #cv2.circle(image, (tapX, tapY), 40, (0, 0, 255), 8) 


#获取屏幕实际的横坐标
def getPosX(col):
    return int(leftMargin+(2*col+1)*chess_piece_radius)

#获取屏幕实际的纵坐标
def getPosY(row):
    return int(topMargin+(2*row+1)*chess_piece_radius)

def showImage(image, scale=1):
    if scale <= 0:
        raise ValueError("缩放比例必须大于0")

    # 调整图像尺寸
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    resized_image = cv2.resize(image, (width, height))

    # 显示图像
    cv2.imshow("chessboard", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def report_matrix_data(matrix):
    if isRed == 1:
         print("红色矩阵数据开始上报")
    else:
        print('黑色矩阵数据开始上报')    
    try:
        if isRed != 1:
            #如果是黑色翻转两次再上报
            matrix = np.rot90(matrix, 2)
        # 将矩阵转换为可序列化的列表
        matrix_list = matrix.tolist()
        jsonText = {'chessboard': matrix_list,'ip':device_ip,'isRed':isRed}
        # 发送 POST 请求上报数据
        response = requests.post("http://localhost:5000/set_chessboard", json=jsonText)
        response.raise_for_status()  # 检查是否有错误状态码
        if isRed == 1:
            print("红色矩阵数据上报成功",jsonText)
        else:
            print('黑方矩阵数据上报成功',jsonText)    
    except requests.exceptions.RequestException as e:
        print("矩阵数据上报失败:", e)


def download_matrix():
    try:
        response = requests.post('http://localhost:5000/get_chessboard',json={'chessboard': chessboard.tolist(),'ip':device_ip,'isRed':isRed})
        response.raise_for_status()  # 检查是否有错误状态码

        data = response.json()
        matrix = np.array(data)
        if isRed != 1:
            matrix = np.rot90(matrix, 2)
        #print("矩阵数据下载成功")
        return matrix
    except requests.exceptions.RequestException as e:
        print("矩阵数据下载失败:", e)

#比较两个棋盘差异并返回
def compare_chessboards(chessboard1, chessboard2):
    diff_indices = np.where(chessboard1 != chessboard2)
    diff_points = list(zip(diff_indices[1], diff_indices[0])) 
    #print(diff_indices)
    print("{} 不同点位 {}".format(diff_points,device_ip))
    return diff_points 

def check_matrices_equal(matrix1, matrix2):
    if matrix1 is None or matrix2 is None:
        return False
    return np.array_equal(matrix1, matrix2)

def isContainRed(x,y,r,outImage):
    # 圆形区域的中心坐标和半径
    center = (x, y)
    radius = r

    # 获取圆形区域的图像副本
    circle_image = outImage[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius].copy()
    # 将图像副本转换为HSV颜色空间
    hsv_circle = cv2.cvtColor(circle_image, cv2.COLOR_BGR2HSV)
    # 定义红色的色调范围
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # 使用颜色阈值分割提取红色区域
    red_mask1 = cv2.inRange(hsv_circle, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_circle, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # 计算红色区域的像素比例
    red_pixel_ratio = np.count_nonzero(red_mask) / (red_mask.shape[0] * red_mask.shape[1])

    # 设置红色像素比例阈值
    red_ratio_threshold = 0.2

    # 判断圆形区域是否包含红色
    if red_pixel_ratio > red_ratio_threshold:
        # print("圆形区域包含红色")
        return True
    else:
        # print("圆形区域不包含红色")
        return False

#服务只负责上报数据 客户端模式只负责调用接口数据如果有差异则进行点操作
current_timestamp = time.time()

#等待别人点击 客户端点击完成开始识别差异 识别到对方已经下棋那么上报这次的数据
WAITE_OTHER = False

# 初始化变量 上次识别的矩阵
server_chessboard = None
last_chessboard = None
coverName = "human_{}.png".format(device_ip.split(":")[0])
leftMargin = 0
topMargin = 0
while True:
    start_time = time.time()
    os.system("adb -s {} shell screencap /sdcard/{}".format(device_ip,coverName))
    os.system("adb -s {} pull /sdcard/{}".format(device_ip,coverName))
    # 读取棋盘图像
    image = cv2.imread(coverName)
    print(f'截图时间',time.time()-start_time)

    if leftMargin==0 and topMargin==0 :

         # 转换为灰度图像
         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

         # 将灰度图像转换为黑白图像
         threshold, black_white = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

         # 进行轮廓检测
         contours, _ = cv2.findContours(black_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

         if len(contours)==0:
             time.sleep(3)
             continue
         # 找到最大的轮廓
         max_contour = max(contours, key=cv2.contourArea)

         # 找到最大轮廓的边界框
         leftMargin, topMargin, w, h = cv2.boundingRect(max_contour)

         #print(f"{leftMargin} {topMargin} {w} {h}")

    
    # 裁剪出棋盘区域
    board_image = image[topMargin:topMargin+h, leftMargin:leftMargin+w]

    # 转换为灰度图像
    gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)

    # 获取棋子宽度
    chess_piece_width = board_image.shape[1] // 9

    #棋子的半径
    chess_piece_radius = chess_piece_width//2

    #获取棋子的高度
    chess_piece_height = board_image.shape[0] // 10

    # 计算最小半径和最大半径
    min_radius = chess_piece_width // 4
    max_radius = chess_piece_width // 3
    start_time = time.time()
    # 进行圆检测
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                            param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)
    # 计算执行时间
    execution_time = time.time() - start_time

    # 输出执行时间
    #print("统计圆形时间 所用时间：", execution_time, "秒")
    if isRed is None:
        #检测(4,9) 位置是否存在0红色
        isRed = isContainRed(getPosX(4),getPosY(9),chess_piece_radius//2,image)
        if isRed:
           isRed = 1
        else:
           isRed = 0   
        print("检测到我方{} 为 红色==> {}".format(device_ip,isRed))

    # 创建棋盘矩阵
    chessboard = np.zeros((10, 9), dtype=str)

    # 如果检测到圆
    if circles is not None:
        # 将检测到的圆转换为对应的坐标并标记在棋盘矩阵中
        for circle in circles[0]:
            x = int(circle[0])
            y = int(circle[1])
            radius = int(circle[2])
            row = int(y / chess_piece_height)
            col = int(x / chess_piece_width)
            #这里检测圆圈范围是否包含红色
            containRed = isContainRed(x,y,radius,board_image)
            if(containRed):
                # 将对应坐标的位置标记为 "R"
                chessboard[row, col] = "R"
            else:
                # 将对应坐标的位置标记为 "B"
                chessboard[row, col] = "B"


    # 遍历棋盘矩阵，将没有棋子的位置标记为 "N"
    for i in range(10):
        for j in range(9):
            if chessboard[i, j] != "R" and chessboard[i, j] != "B":
                chessboard[i, j] = "N"
    # 打印棋盘矩阵
    #for row in chessboard:
    #    print(" ".join(row))

    if np.all(chessboard == 'N'):     
        print("暂未识别到棋子")
        continue

    if check_matrices_equal(last_chessboard,chessboard):
        print("本地数据无变化无需上报")
    else:
       #上报本机数据
       report_matrix_data(chessboard)
       last_chessboard = chessboard.copy()

    #下载服务端的矩阵
    server_chessboard = download_matrix()
    
    if(server_chessboard is None or np.all(server_chessboard == 'N')):
        print(("{}:服务器暂无数据").format(device_ip))
        continue

    if check_matrices_equal(chessboard,server_chessboard):
        print("{}:服务器数据与本地数据相同等待新数据".format(device_ip))
        continue

    # 找到不同的位置
    diff_indices = compare_chessboards(chessboard,server_chessboard)
    if diff_indices is None or len(diff_indices)!=2:
        print("{}:服务器数据与本地数据不同但点位有问题".format(device_ip))
        continue

    # 输出不同位置的坐标
    for i in range(2):
        row = diff_indices[i][1]
        col = diff_indices[i][0]
        tapPos(row,col)
    for i in range(2):
        row = diff_indices[1-i][1]
        col = diff_indices[1-i][0]
        tapPos(row,col) 
    print("点击总时间{}".format(time.time() - start_time))
    # 休眠2秒
    #time.sleep(2)