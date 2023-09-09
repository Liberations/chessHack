import subprocess
import re
from functools import partial
import time
import subprocess
import platform

def run_thread(ip):
    print("收到IP",ip)
    subprocess.Popen(['python', 'chess_client.py', ip])

def get_adb_devices():
    # 运行 adb devices 命令
    result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
    
    # 获取命令输出
    output = result.stdout
    
    # 解析输出并提取IP地址和端口号
    pattern = r'(\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b):(\d+)'
    matches = re.findall(pattern, output)
    
    return matches

while True:

    # 获取设备列表中的IP地址
    ip_list = get_adb_devices()
    if len(ip_list) < 2 :
        print("至少保证两台在线")
        time.sleep(5)
        continue
    #192.168.101.16 iqoo7
    #192.168.101.15 iqoo11
    # 遍历IP地址列表，为每个IP地址启动线程
    for i in range(2):
        ip,port = ip_list[i]
        print("启动IP",ip)
        ip = ip +":" +port
        # 使用 functools.partial 创建一个新的函数，并将 ip 和 port 参数预先绑定
        partial_run_thread = partial(run_thread, ip)

        # 手动调用新创建的函数
        partial_run_thread()
    break    