import cv2
import numpy as np
import time

def showImage(image, scale=1):
    if scale <= 0:
        raise ValueError("缩放比例必须大于0")

    # 调整图像尺寸
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    resized_image = cv2.resize(image, (width, height))

    # 显示图像
    cv2.imshow("棋盘截图", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 初始化变量 上次识别的矩阵
previous_chessboard = None

while True:
    # 读取棋盘图像
    image = cv2.imread("chessboard1.jpg")

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 将灰度图像转换为黑白图像
    threshold, black_white = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 进行轮廓检测
    contours, _ = cv2.findContours(black_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)

    # 找到最大轮廓的边界框
    x, y, w, h = cv2.boundingRect(max_contour)

    print(f"{x} {y} {w} {h}")

    # 裁剪黑白图像
    cropped_image = black_white[y:y+h, x:x+w]

    # 进行边缘检测
    edges = cv2.Canny(gray, 80, 160)

    # 显示带有圆的棋盘图像
    #showImage(cropped_image,scale=0.3)

    # 裁剪出棋盘区域
    board_image = image[y:y+h, x:x+w]

    # 转换为灰度图像
    gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)

    # 获取棋子宽度
    chess_piece_width = board_image.shape[1] // 9

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

            # 假设棋盘格子的宽度为 w，高度为 h
            w = board_image.shape[1] // 9
            h = board_image.shape[0] // 10

            row = int(y / h)
            col = int(x / w)

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
     # 如果上一次的棋盘不为空，则比较差异
    if previous_chessboard is not None:
        # 找到不同的位置
        diff_indices = np.where(chessboard != previous_chessboard)

        # 输出不同位置的坐标
        for i in range(len(diff_indices[0])):
            row = diff_indices[0][i]
            col = diff_indices[1][i]
            print("找到差异 ({}, {})".format(row, col))

    # 更新上一次的棋盘
    previous_chessboard = chessboard.copy()

    # 休眠2秒
    time.sleep(2)