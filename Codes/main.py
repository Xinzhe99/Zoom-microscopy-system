# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

import torch
from PyQt5 import QtCore
import sys
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication
import main_window


if __name__ == '__main__':

    #适应桌面分辨率
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    app = QApplication(sys.argv)

    # 设置全局字体
    font = QFont("Times New Roman")
    font.setStyleHint(QFont.Serif)
    font.setPointSize(9)
    QApplication.setFont(font)

    my_window = main_window.MyWindow()
    my_window.ui.show()



    sys.exit(app.exec_())

