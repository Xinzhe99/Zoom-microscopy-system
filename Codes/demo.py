import time
from PyQt5.QtCore import QThread

class MyThread(QThread):
    def __init__(self):
        super().__init__()

    def run(self):
        print("Thread Started")
        time.sleep(5)
        print("Thread Finished")

if __name__ == '__main__':
    thread = MyThread()
    thread.start() # 手动启动线程
    thread.wait() # 等待线程完成

    print("Main Program Exited")