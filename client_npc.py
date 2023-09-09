import cv2
import numpy as np
import time
import os
import requests

#iqoo11
device_ip = "192.168.101.15:40471"

#通过矩阵的x,y计算实际点击的位置
def tapPos(row,col):
    tapX = leftMargin+int((2*col+1)*chess_piece_radius)
    tapY = topMargin+int((2*row+1)*chess_piece_radius)
    os.system("adb -s {} shell input tap {} {}".format(device_ip,tapX,tapY))
    print("点击位置{},{} {},{}".format(row,col,tapX,tapY))
    cv2.circle(image, (tapX, tapY), 40, (0, 0, 255), 8)
    # 实际(2,1) (4,3)  期望 (4，7) (6，9)  实际返回的 (7, 4), (9, 6)  



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
    try:
        # 将矩阵转换为可序列化的列表
        matrix_list = matrix.tolist()

        # 发送 POST 请求上报数据
        response = requests.post("http://localhost:5000/set_chessboard", json={'chessboard': matrix_list,'ip':device_ip})
        response.raise_for_status()  # 检查是否有错误状态码

        print("矩阵数据上报成功")
    except requests.exceptions.RequestException as e:
        print("矩阵数据上报失败:", e)


def download_matrix():
    try:
        response = requests.get('http://localhost:5000/get_chessboard?ip={}'.format(device_ip))
        response.raise_for_status()  # 检查是否有错误状态码

        data = response.json()
        matrix = np.array(data)
        print("矩阵数据下载成功")
        return matrix
    except requests.exceptions.RequestException as e:
        print("矩阵数据下载失败:", e)

#比较两个棋盘差异并返回
def compare_chessboards(chessboard1, chessboard2):
    diff_indices = np.where(chessboard1 != chessboard2)
    diff_points = list(zip(diff_indices[1], diff_indices[0])) 
    print(diff_indices)
    print(diff_points)
    return diff_points 

#服务只负责上报数据 客户端模式只负责调用接口数据如果有差异则进行点操作
current_timestamp = time.time()

#等待别人点击 客户端点击完成开始识别差异 识别到对方已经下棋那么上报这次的数据
WAITE_OTHER = False

# 初始化变量 上次识别的矩阵
server_chessboard = None
coverName = "human_{}.png".format(device_ip.replace(".", "_").replace(":", "_"))
leftMargin = 0
topMargin = 0
while True:
    os.system("adb -s {} shell screencap /sdcard/{}".format(device_ip,coverName))
    os.system("adb -s {} pull /sdcard/{}".format(device_ip,coverName))
    # 读取棋盘图像
    image = cv2.imread(coverName)

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

         print(f"{leftMargin} {topMargin} {w} {h}")

    # 裁剪黑白图像
    #cropped_image = black_white[topMargin:topMargin+h, leftMargin:leftMargin+w]



    # 显示带有圆的棋盘图像
    #showImage(cropped_image,scale=0.3)

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

    # 进行圆检测
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                            param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)

    # 创建棋盘矩阵
    chessboard = np.zeros((10, 9), dtype=str)

    # 如果检测到圆
    if circles is not None:
        # 将检测到的圆转换为对应的坐标并标记在棋盘矩阵中
        for circle in circles[0]:
            x = int(circle[0])
            y = int(circle[1])
            row = int(y / chess_piece_height)
            col = int(x / chess_piece_width)

            # 将对应坐标的位置标记为 "Y"
            chessboard[row, col] = "Y"

    # 遍历棋盘矩阵，将没有棋子的位置标记为 "N"
    for i in range(10):
        for j in range(9):
            if chessboard[i, j] != "Y":
                chessboard[i, j] = "N"
    # 打印棋盘矩阵
    for row in chessboard:
        print(" ".join(row))

    if np.all(chessboard == 'N'):     
        print("暂未识别到棋子")
        continue

    if WAITE_OTHER:
       diff_indices = compare_chessboards(chessboard,server_chessboard)
       if diff_indices is None or len(diff_indices)!=2:
            server_chessboard = download_matrix()
            continue
       print("对方已经落子开始上报")
       report_matrix_data(chessboard)
       WAITE_OTHER = False
       server_chessboard = chessboard.copy
       #等待服务器响应
       #time.sleep(3)
    else:
       server_chessboard = None
       #下载服务端的矩阵
       server_chessboard = download_matrix()
       if(server_chessboard is None or np.all(server_chessboard == 'N')):
           print("服务器暂无数据")
           report_matrix_data(chessboard)
           continue
       # 如果上一次的棋盘不为空，则比较差异
       if server_chessboard is not None:
           for row in chessboard:
                print(" ".join(row))
           # 找到不同的位置
           diff_indices = compare_chessboards(chessboard,server_chessboard)
           if diff_indices is None or len(diff_indices)!=2:
               continue
           # 输出不同位置的坐标
           print("收到机器人已落子")
           for i in range(2):
               row = diff_indices[i][1]
               col = diff_indices[i][0]
               tapPos(row,col)
               time.sleep(0.3)  
           for i in range(2):
               row = diff_indices[1-i][1]
               col = diff_indices[1-i][0]
               tapPos(row,col)
               time.sleep(0.3)     
           #标记客户端点击完成 开始等待对方下子    
           WAITE_OTHER = True
           #showImage(image,0.3)
           #休眠一会等对方下子    
           #time.sleep(3)

               
    
    # 休眠2秒
    time.sleep(2)