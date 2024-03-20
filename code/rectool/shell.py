__all__ = ['subprocess_popen', 'choose_gpu']
import numpy as np
import subprocess

def subprocess_popen(statement):
    """
    执行shell命令，并执行结果，可以判断执行是否成功
    """
    p = subprocess.Popen(statement, shell=True, stdout=subprocess.PIPE)  # 执行shell语句并定义输出格式
    while p.poll() is None:  # 判断进程是否结束（Popen.poll()用于检查子进程（命令）是否已经执行结束，没结束返回None，结束后返回状态码）
        if p.wait() != 0:  # 判断是否执行成功（Popen.wait()等待子进程结束，并返回状态码；如果设置并且在timeout指定的秒数之后进程还没有结束，将会抛出一个TimeoutExpired异常。）
            print("命令执行失败，请检查设备连接状态")
            return False
        else:
            re = p.stdout.readlines()  # 获取原始执行结果
            result = []
            for i in range(len(re)):  # 由于原始结果需要转换编码，所以循环转为utf8编码并且去除\n换行
                res = re[i].decode('utf-8').strip('\r\n')
                result.append(res)
            # print(result)
            return result

def choose_gpu():
    """
    根据内存选择最优gpu
    """
    shell = "nvidia-smi -q -d Memory |grep -A5 GPU|grep Free"     # -A5参数根据实际输出修改为-A4或其他
    res = subprocess_popen(shell)
    res = [ x.split(':')[-1].strip() for x in res ]
    res = [ int(x.split(' ')[0]) for x in res]
    # print(res)
    max_free = np.argmax(res)
    # print(max_free)
    return max_free 

# print(choose_gpu())