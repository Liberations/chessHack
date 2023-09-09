from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

chessList = [None,None]

#比较两个棋盘差异并返回
def compare_chessboards(chessboard1, chessboard2):
    diff_indices = np.where(chessboard1 != chessboard2)
    diff_points = list(zip(diff_indices[1], diff_indices[0])) 
    #print(diff_indices)
    print("找到不同点位{}".format(diff_points))
    return diff_points 

def printChessBoard(chessboard):
    if chessboard is not None:
        for row in chessboard:
            print(" ".join(row))

@app.route('/set_chessboard', methods=['POST'])
def set_chessboard():
    isRed = request.json.get('isRed')
    global chessList
    if isRed == 1:
       redChessBoard = np.array(request.json.get('chessboard'))
       chessList[0] = redChessBoard
       print("收到红方矩阵数据\n",redChessBoard)
    else:
       blackChessBoard = np.array(request.json.get('chessboard'))
       chessList[1] = blackChessBoard
       print("收到黑方矩阵数据\n",blackChessBoard)
    print("此时红色棋子")
    printChessBoard(chessList[0])
    print("此时黑色棋子")
    printChessBoard(chessList[1])
    return '数据上报成功'

@app.route('/get_chessboard', methods=['POST'])
def get_chessboard():
    isRed = request.json.get('isRed')
    global chessList
    # 在这里将ip保存到服务器端，可以将其存储到数据库或使用其他方式进行处理
    res = None
    if isRed == 1:
        print("红方请求到的矩阵")
        matrix = chessList[1]
    else:
        print("黑方请求到的矩阵")
        matrix = chessList[0]
    if matrix is not None:
        res = matrix.tolist() 
    else:
        res = np.full((10, 9), 'N', dtype=str).tolist()    
   
    printChessBoard(matrix)
    return jsonify(res)

@app.route('/clean_chessboard', methods=['GET'])
def clean_chessboard():
    global redChessBoard 
    global blackChessBoard
    redChessBoard = np.full((10, 9), 'N', dtype=str)
    blackChessBoard = np.full((10, 9), 'N', dtype=str)
    global chessList
    chessList = [redChessBoard,blackChessBoard]
    return '清空完成'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)