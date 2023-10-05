# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

import re
import PySpin
from PyQt5.QtGui import QImage, QPixmap, QIcon
import os
from PyQt5.QtCore import QTimer, QThread, Qt, QRect, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QInputDialog
from tools import is_number,cal_clarity,max_y,is_scale_valid
import cv2
from lens import Lens
from PyQt5.QtWidgets import QApplication, QWidget
import time
import sys
from PyQt5 import uic
import numpy as np
from scipy.optimize import curve_fit
import serial.tools.list_ports
import fusion
from torch.cuda import is_available as gpu_is_available

#允许多个进程同时使用OpenMP库
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#定义支持的图片格式
support_format=[".jpg", ".png", ".bmp"]

# flag定义全局变量 图片保存的序号默认从1开始
flag_save_number = 1
# 检查屈光度更新的flag
diop_updata_flag=10000

#局部对焦flag
roi_flag=False

#相机是否开了flag
flag_cam_is_on = False



class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        if getattr(sys,'frozen', False):
            ui_path = os.path.join(sys._MEIPASS, 'design_menu.ui')
        else:
            ui_path = 'design_menu.ui'

        self.ui = uic.loadUi(ui_path)

        if getattr(sys, 'frozen', False):
            ico_path = os.path.join(sys._MEIPASS, 'myico.ico')
        else:
            ico_path = 'myico.ico'
        self.ui.setWindowIcon(QIcon(ico_path))

        #一开始没有连接任何串口
        self.port_checked=''

        #启动按钮
        btn_start_capture=self.ui.btn_begin_caputre #获取相机画面按钮
        btn_start_capture.clicked.connect(self.start_capture)

        #停止按钮
        btn_stop_capture=self.ui.btn_stop_caputre#停止获取画面的按钮
        btn_stop_capture.clicked.connect(self.stop_capture)
        btn_stop_capture.setEnabled(False)

        # 显示图像的标签
        self.img_show_lable = self.ui.video_label
        # 创建一个QTimer对象
        self.timer_video = QTimer()
        # 连接定时器的timeout信号到update_image槽函数
        self.timer_video.timeout.connect(self.update_image)
        # 创建一个QTimer对象用于更新曝光时间的显示
        self.exp_show_timer = QTimer()
        # 连接定时器的timeout信号到槽函数
        self.exp_show_timer.timeout.connect(self.update_exp_label)
        # 创建一个QTimer对象用于更新gain的显示
        self.gain_show_timer = QTimer()
        # 连接定时器的timeout信号到槽函数
        self.gain_show_timer.timeout.connect(self.update_gain_label)
        # 创建一个QTimer对象用于更新fps的显示
        self.fps_show_timer = QTimer()
        # 连接定时器的timeout信号到槽函数
        self.fps_show_timer.timeout.connect(self.update_fps_label)
        # 创建一个QTimer对象用于保存图片
        self.img_save_timer = QTimer()
        # 连接定时器的timeout信号到槽函数
        self.img_save_timer.timeout.connect(self.save_one_img)
        # 显示相机信息的标签
        self.lb_video_inf = self.ui.video_inf
        # 显示相机数量的标签
        self.lb_num_cam = self.ui.video_num

        #获取显示lable的width和height
        self.show_width = self.ui.video_label.geometry().width()
        self.show_height = self.ui.video_label.geometry().height()
        # 绑定自动曝光按钮
        self.btn_auto_exposure=self.ui.btn_Auto_Exposure
        # 自动曝光初始化默认开启
        self.btn_auto_exposure.setChecked(True)
        # 该按钮默认不能用，开始捕获图像后才可以
        self.btn_auto_exposure.setEnabled(False)

        #修改曝光模式
        self.btn_auto_exposure.stateChanged.connect(self.change_exposure_mode)
        self.exposure_time_slider=self.ui.exp_control#绑定手动调节曝光的slider
        self.exposure_time_slider.setRange(6, 30000.0)#初始化设置slider的上下限，不能删，删了报错
        self.exposure_time_slider.valueChanged.connect(self.set_exposure)#绑定曝光时间设置函数
        self.exposure_time_slider.setEnabled(False)  # 初始化冻结slider
        self.lb_exp_show=self.ui.exp_show#显示曝光时间

        # 绑定自动gain按钮
        self.btn_auto_gain = self.ui.btn_Auto_Gain  # 绑定gain按钮
        # 自动gain默认开启
        self.btn_auto_gain.setChecked(True)  # 自动gain初始化默认开启
        # 该按钮默认不能用，开始捕获图像后才可以
        self.btn_auto_gain.setEnabled(False)

        #修改gain模式
        self.btn_auto_gain.stateChanged.connect(self.change_gain_mode)
        self.gain_slider = self.ui.gain_control  # 绑定手动调节gain的slider
        self.gain_slider.setRange(0, 47.99)  # 初始化设置slider的上下限，不能删，删了会因为没有初值，下面代码报错，实际还会重新根据实际情况定新的值
        self.gain_slider.valueChanged.connect(self.set_gain)  # 绑定曝光时间设置函数
        self.gain_slider.setEnabled(False)# 初始化冻结slider
        self.lb_gain_show = self.ui.gain_show  # 绑定显示gain值

        #链接用于保存/手动停止保存图片的按钮
        self.btn_save_pictures=self.ui.btn_begin_save
        self.btn_save_pictures.clicked.connect(self.start_save_picture)
        self.btn_stop_save_pictures = self.ui.btn_stop_save
        self.btn_stop_save_pictures.clicked.connect(self.stop_save_picture)

        #用于保存图像的flag,默认情况不保存
        self.need_save_fig=False

        #初始情况下冻结以下按钮
        self.btn_save_pictures.setEnabled(False)
        self.btn_stop_save_pictures.setEnabled(False)

        #连接透镜按钮
        self.btn_connect_lens=self.ui.btn_begin_connect_len
        self.btn_connect_lens.clicked.connect(self.create_len)

        # 连接透镜串口刷新按钮
        self.btn_connect_lens = self.ui.btn_com_refresh
        self.btn_connect_lens.clicked.connect(self.refresh_ports)

        #设置屈光度按钮
        self.btn_to_set_diopter=self.ui.btn_set_diopter
        self.btn_to_set_diopter.clicked.connect(self.setdiopter)

        # 重置屈光度按钮
        self.btn_reset_len=self.ui.btn_Reset_diopter
        self.btn_reset_len.clicked.connect(self.reset_diopter)

        #屈光度循环开启按钮
        self.btn_begin_cycle=self.ui.btn_cycle_diopter
        self.btn_begin_cycle.clicked.connect(self.begin_cycle)

        #屈光度循环关闭按钮
        self.btn_stop_cycle=self.ui.btn_stop_cycle_diopter
        self.btn_stop_cycle.clicked.connect(self.stop_cycle)


        #用于绑定cycle用的编辑框，监听是否发生改变，计算图片数量用
        self.max_diopter_input=self.ui.max_diopter_input
        self.min_diopter_input=self.ui.min_diopter_input
        self.step_diopter_input=self.ui.step_diopter_input
        # 监听这两个edit,按下回车键或者失去焦点时触发，检查计算
        self.max_diopter_input.editingFinished.connect(self.updade_count_number)
        self.min_diopter_input.editingFinished.connect(self.updade_count_number)
        self.step_diopter_input.editingFinished.connect(self.updade_count_number)

        #绑定手动输入曝光和增益的按钮和编辑框初始化无法使用
        self.exp_edit=self.ui.exp_edit
        self.btn_exp_set=self.ui.exp_set
        self.gain_edit=self.ui.gain_edit
        self.btn_gain_set=self.ui.gain_set
        self.exp_edit.setEnabled(False)
        self.btn_exp_set.setEnabled(False)
        self.gain_edit.setEnabled(False)
        self.btn_gain_set.setEnabled(False)

        #绑定检查曝光时间和增益的两个函数
        self.exp_edit.editingFinished.connect(self.check_edit_exp)
        self.gain_edit.editingFinished.connect(self.check_edit_gain)

        #绑定设置曝光时间和增益的按钮
        self.btn_exp_set.clicked.connect(self.set_exposure_edit)
        self.btn_gain_set.clicked.connect(self.set_gain_edit)

        #绑定用于自动对焦的按钮
        self.btn_auto_focus=self.ui.btn_autofocus
        self.btn_auto_focus.clicked.connect(self.auto_focus)

        # 获取端口列表
        self.get_port_list()

        #绑定录像按钮
        self.btn_start_capture_video=self.ui.btn_begin_save_2
        self.btn_start_capture_video.clicked.connect(self.start_recording)

        #设置录制的flag,默认不录制
        self.record_flag = False

        #设置录像的帧列表
        self.frameslist=[]

        #绑定结束录像按钮
        self.btn_stop_capture_video = self.ui.btn_stop_save_2
        self.btn_stop_capture_video.clicked.connect(self.stop_recording)

        #防止误点，一开始冻结录像按钮
        self.btn_stop_capture_video.setEnabled(False)
        self.btn_start_capture_video.setEnabled(False)

        # 冻结透镜区域功能按钮
        self.ui.btn_set_diopter.setEnabled(False)
        self.ui.btn_Reset_diopter.setEnabled(False)
        self.ui.btn_cycle_diopter.setEnabled(False)
        self.ui.btn_stop_cycle_diopter.setEnabled(False)

        #冻结保存图像和图片的停止按钮
        self.ui.btn_stop_save.setEnabled(False)
        self.ui.btn_stop_save_2.setEnabled(False)

        #绑定双击局部聚焦
        self.ui.video_label.mouseDoubleClickEvent = self.roi_focus  # 绑定双击信号

        #绑定
        self.ui.btn_set_focus.clicked.connect(self.set_focus)

        #设置默认的对焦参数
        self.roi_height = 50
        self.roi_width = 50

        #框选区域进行自动对焦
        self.start_pos = None
        self.end_pos = None

        #绑定框选对焦
        self.ui.video_label.mousePressEvent = self.boxout_mousePressEvent
        self.ui.video_label.mouseReleaseEvent = self.boxout_mouseReleaseEvent
        self.ui.video_label.mouseMoveEvent = self.boxout_mouseMoveEvent

        # 鼠标移动中Flag,防止两个鼠标事件起冲突，效果和过滤器差不多
        self.move_flag = False

        # 创建一个QTimer对象来检测串口
        self.timer_port = QTimer()
        # 连接定时器的timeout信号
        self.timer_port.timeout.connect(self.check_port_status)

        #绑定自动搜寻按钮
        self.ui.btn_autoSearch.clicked.connect(self.auto_search)
        self.ui.btn_autoSearch.setEnabled(False)
        self.cal_auto_search = False
        self.frame_list_cal=[]

        #绑定融合图像栈的按钮
        self.btn_fusion = self.ui.btn_fusion_stack
        self.btn_fusion.clicked.connect(self.fusion_stack)

        # 绑定选择融合图像栈路径的按钮
        self.btn_stack_path = self.ui.btn_stack_path
        self.btn_stack_path.clicked.connect(self.browse_stack_dir)

        # 绑定选择融合图像栈融合结果保存路径的按钮
        self.btn_save_result_path = self.ui.btn_save_result_path
        self.btn_save_result_path.clicked.connect(self.browse_fusion_result_dir)

        #检查gpu是否可以用
        if gpu_is_available():
            self.ui.btn_use_GPU.setChecked(True)
        else:
            self.ui.btn_use_GPU.setChecked(False)

        #自动对焦时候所用的计算清晰度的计算时间宏定义(s)
        self.cal_sleep=1

        #一开始不可以用自动对焦，连上相机和透镜才可以
        self.ui.btn_autofocus.setEnabled(False)

    #获取端口列表
    def get_port_list(self):
        plist = list(serial.tools.list_ports.comports())
        if len(plist) <= 0:
            QMessageBox.information(self, 'Notice', 'Cannot find any com can be used!')
        else:
            self.ui.comboBox_port.clear()
            for i in range(0, len(plist)):
                self.plist_0 = list(plist[i])
                self.ui.comboBox_port.addItem(str(self.plist_0[0]))
    # 检测连接函数
    def check_port_status(self):
        #每隔一秒钟获取一次port列表
        plist = list(serial.tools.list_ports.comports())
        port_list=[]
        for i in range(0, len(plist)):
            plist_0 = list(plist[i])
            port_list.append(str(plist_0[0]))
        #串口拔出了
        if not self.port_checked in port_list:
            QMessageBox.information(self, 'Notice', 'Lens connection disconnected!')
            self.ui.lb_show_com.setText('Not connected')
            self.ui.lb_temp_show.setText('')
            self.ui.lb_firmware_show.setText('')
            self.ui.lb_posi_dip_show.setText('')
            self.ui.lb_neg_dip_show.setText('')
            self.ui.lb_lens_SN_show.setText('')
            self.refresh_ports()
            self.timer_port.stop()
        else:
            self.ui.lb_show_com.setText('Connected')
    #刷新串口列表
    def refresh_ports(self):
        self.lens.close_connection()
        self.ui.btn_autoSearch.setEnabled(False)
        self.get_port_list()
        self.port_checked = ''
    # 用于手动设置编辑框输入的曝光值
    def set_exposure_edit(self):
        exp_to_set_text = self.exp_edit.text()
        if is_number(exp_to_set_text):
            exp_to_set=float(exp_to_set_text)
            self.cam.ExposureTime.SetValue(exp_to_set)
            self.exposure_time_slider.setValue(exp_to_set)
        else:
            QMessageBox.information(self, "notice", 'Please enter exposure value')

    # 用于手动设置编辑框输入的增益值
    def set_gain_edit(self):
        gain_to_set_text=self.gain_edit.text()
        if is_number(gain_to_set_text):
            gain_to_set = float(gain_to_set_text)
            self.cam.Gain.SetValue(gain_to_set)
            self.gain_slider.setValue(gain_to_set)
        else:
            QMessageBox.information(self, "notice", 'Please enter gain value')


    #用于初始化相机并且开启定时器捕获图像
    def start_capture(self):
        try:

            self.camera_system = PySpin.System.GetInstance()
            self.cam_list = self.camera_system.GetCameras()
            self.cam = self.cam_list[0]
            # 获取相机数量,这里只支持一个!
            num_cameras = self.cam_list.GetSize()
            if num_cameras != 0:
                self.nodemap_tldevice = self.cam.GetTLDeviceNodeMap()#获取跟传输有关的属性

                deviceFeaturesInfo = PySpin.CCategoryPtr(self.nodemap_tldevice.GetNode('DeviceInformation')).GetFeatures()
                self.modelName = PySpin.CValuePtr(deviceFeaturesInfo[3]).ToString()#相机型号
                print(self.modelName[-1])
                serialNum = PySpin.CValuePtr(deviceFeaturesInfo[1]).ToString()#相机序列号
                self.cam.Init()

                self.nodemap = self.cam.GetNodeMap()#获取所有属性

                # Set acquisition mode to continuous
                node_acquisition_mode = PySpin.CEnumerationPtr(self.nodemap.GetNode('AcquisitionMode'))
                if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                    print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                    return False
                # Retrieve entry node from enumeration node
                node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
                if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                        node_acquisition_mode_continuous):
                    print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                    return False

                acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
                node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
                print('Acquisition mode set to continuous...')

                # 获取相机版本型号
                self.lb_video_inf.setText(self.modelName)

                # 使能按钮
                self.btn_save_pictures.setEnabled(True)
                self.btn_stop_save_pictures.setEnabled(True)
                self.btn_auto_exposure.setEnabled(True)
                self.btn_auto_gain.setEnabled(True)
                self.ui.btn_stop_caputre.setEnabled(True)
                self.lb_num_cam.setText('%d' % num_cameras)
                self.btn_stop_capture_video.setEnabled(False)
                self.btn_start_capture_video.setEnabled(True)

                if self.ui.lb_show_com.text() == 'Connected':
                    self.ui.btn_autoSearch.setEnabled(True)
                    self.ui.btn_autofocus.setEnabled(True)

                #相机开始获取图像
                self.cam.BeginAcquisition()
                print('All initial finished,Begin to Acquisition...')

                #获取实时fps
                self.real_fps = self.cam.AcquisitionResultingFrameRate.GetValue()

                #更新标签
                self.update_exp_label()
                self.update_gain_label()

                # 相机初始化自动曝光
                self.Reset_exposure()
                auto_exposure_value=self.cam.ExposureTime.GetValue()
                self.exposure_time_slider.setValue(auto_exposure_value)

                # 相机初始化增益
                self.Reset_gain()
                auto_gain_value = self.cam.Gain.GetValue()
                self.ui.gain_control.setValue(auto_gain_value)

                #用定时器来获取新的图像
                self.timer_video.start(0)

                # 用定时器来更新曝光时间显示的标签
                self.exp_show_timer.start(1000)

                # 用定时器来更新gain显示的标签
                self.gain_show_timer.start(1000)

                # 用定时器来获取fps
                self.fps_show_timer.start(1000)

                #关闭这个按钮，不然会报错
                self.ui.btn_begin_caputre.setEnabled(False)

                # 支持的曝光时间非常久，容易卡，这里手动设置为30000
                self.exposure_time_slider.setRange(self.cam.ExposureTime.GetMin(), 30000)

                #设置增益的slider上下限
                self.gain_slider.setRange(self.cam.Gain.GetMin(),self.cam.Gain.GetMax())

                self.ui.btn_stop_save_2.setEnabled(False)
                self.ui.btn_stop_save.setEnabled(False)


            else:
                QMessageBox.information(self, "notice", 'Please check if the camera is connected')
        except:
            QMessageBox.information(self, "notice", 'Please check if the camera is connected')
    #用于虚假关闭相机，其实还在
    def stop_capture(self):
        lb_video_inf = self.ui.video_inf
        lb_video_inf.setText('EndAcquisition!')
        self.ui.video_num.setText('0')
        self.timer_video.stop()
        self.cam.EndAcquisition()
        #控制按钮，防止误点崩溃
        self.ui.btn_begin_caputre.setEnabled(True)
        self.ui.btn_stop_caputre.setEnabled(False)
        self.img_show_lable.clear()

        self.ui.btn_begin_save.setEnabled(False)
        self.ui.btn_begin_save_2.setEnabled(False)
        self.ui.btn_stop_save.setEnabled(False)
        self.ui.btn_stop_save_2.setEnabled(False)
        self.ui.btn_autoSearch.setEnabled(False)
        self.ui.btn_autofocus.setEnabled(False)

        #关闭三个定时器，停止更新
        self.exp_show_timer.stop()
        self.gain_show_timer.stop()
        self.fps_show_timer.stop()
        self.ui.lb_show_fps.setText('')
        self.ui.exp_show.setText('')
        self.ui.gain_show.setText('')



        # 自动曝光初始化默认开启
        self.btn_auto_exposure.setChecked(True)
        # 该按钮默认不能用，开始捕获图像后才可以
        self.btn_auto_exposure.setEnabled(False)
        # 自动gain默认开启
        self.btn_auto_gain.setChecked(True)  # 自动gain初始化默认开启
        # 该按钮默认不能用，开始捕获图像后才可以
        self.btn_auto_gain.setEnabled(False)
        print('End EndAcquisition!')


    #用于定时器获取一张图像并显示出来
    def update_image(self):

        # 根据实时曝光时间调整等待时间
        timeout = 1000
        self.real_exp_time = self.cam.ExposureTime.GetValue()
        if self.cam.ExposureTime.GetAccessMode() == PySpin.RW or self.cam.ExposureTime.GetAccessMode() == PySpin.RO:
            # The exposure time is retrieved in µs so it needs to be converted to ms to keep consistency with the unit being used in GetNextImage
            timeout = (int)(self.real_exp_time / 1000 + 1000)

        image_result_ori = self.cam.GetNextImage(timeout)

        #判断是否是彩色相机,根据相机型号调整image_result
        if self.modelName[-1]=='C':
            image_result_color=image_result_ori.Convert(PySpin.PixelFormat_BGR8, PySpin.HQ_LINEAR)
            image_data_np = image_result_color.GetNDArray()
        elif self.modelName[-1]=='M':
            image_result_mon=image_result_ori.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
            image_data_np = image_result_mon.GetNDArray()
        else:
            pass

        # #检查完整性
        # if image_result_ori.IsIncomplete():
        #     print('Image incomplete with image status %d ...' % image_result_ori.GetImageStatus())

        # 显示(灰度)
        if self.modelName[-1] == 'M':
            image_pixmap = QPixmap.fromImage(QImage(image_data_np.data, image_data_np.shape[1], image_data_np.shape[0],
                                                    QImage.Format_Grayscale8))
            self.img_show_lable.setPixmap(image_pixmap)
            # 调整尺寸
            self.img_show_lable.setScaledContents(True)

        # 显示(RGB)
        elif self.modelName[-1] == 'C':
            image_pixmap = QPixmap.fromImage(QImage(image_data_np.data, image_data_np.shape[1], image_data_np.shape[0],
                                                    QImage.Format_BGR888))
            self.img_show_lable.setPixmap(image_pixmap)
            # 调整尺寸
            self.img_show_lable.setScaledContents(True)


        #如果同时需要保存下来,标志位由定时器控制,每x ms 置一次True,保存一张图片
        if self.need_save_fig == True:
            #开个子线程 保存图片 不然会拖慢进程 影响显示
            #判断是否手动选择了需要的格式，如果没有默认jpg
            format_need=self.ui.comboBox_output_format_2.currentText()

            self.save_img_sub_thread=Save_img(data=image_result_ori,path=self.pic_save_path,save_format=format_need)

            self.need_save_fig=False#保存flag置为False,等待下一次置为True

        #如果需要录制下来
        if self.record_flag==True:
            self.frameslist.append(image_result_ori)

        #如果是在自动搜寻清晰的区域的上下平面
        if self.cal_auto_search==True:
            if len(image_data_np.shape) == 3:
                gray = cv2.cvtColor(image_data_np, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_data_np
            gray=cv2.resize(gray,(800,550))
            self.frame_list_cal.append(gray)
            self.cal_auto_search=False
        # 释放内存
        image_result_ori.Release()
        return image_data_np

    #用于重置相机曝光模式到自动
    def Reset_exposure(self):
        if self.cam.ExposureAuto.GetAccessMode() != PySpin.RW:
            print('Unable to enable automatic exposure (node retrieval). Non-fatal error...')
        self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)

        #启动定时器来检测实时的曝光值 并且同步到slider上
        # self.exposure_time_slider.setValue(self.cam.ExposureTime.GetValue())
        print('Automatic exposure enabled...')

    #用于相机曝光时间修改
    def set_exposure(self):
        self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        if self.cam.ExposureTime.GetAccessMode() != PySpin.RW:
            print('Unable to set exposure time. Aborting...')

        exposure_time_to_set = self.exposure_time_slider.value()
        self.cam.ExposureTime.SetValue(exposure_time_to_set)


    # 用于更新曝光时间显示的标签
    def update_exp_label(self):
        # 获取实时曝光
        self.real_exp_time = self.cam.ExposureTime.GetValue()# us
        self.lb_exp_show.setText(str(self.real_exp_time))

    #用于相机曝光模式改变
    def change_exposure_mode(self):
        if self.btn_auto_exposure.isChecked():

            self.Reset_exposure()
            self.exposure_time_slider.setEnabled(False)
            self.exp_edit.setEnabled(False)
            self.btn_exp_set.setEnabled(False)

        else:
            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            self.cam.ExposureTime.SetValue(self.exposure_time_slider.value())
            self.exposure_time_slider.setEnabled(True)
            self.exp_edit.setEnabled(True)
            self.btn_exp_set.setEnabled(True)

    #用于gain模式改变
    def change_gain_mode(self):
        #自动gain开启
        if self.btn_auto_gain.isChecked():
            self.Reset_gain()
            self.gain_slider.setEnabled(False)
            self.gain_edit.setEnabled(False)
            self.btn_gain_set.setEnabled(False)
        # 自动gain关闭
        else:
            self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
            self.cam.Gain.SetValue(self.gain_slider.value())
            self.gain_slider.setEnabled(True)
            self.gain_edit.setEnabled(True)
            self.btn_gain_set.setEnabled(True)

    #用于手动设置gain
    def set_gain(self):
        if self.cam.Gain.GetAccessMode() != PySpin.RW:
            print('Unable to set gain. Aborting...')
        self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        gain_to_set = self.gain_slider.value()
        self.cam.Gain.SetValue(gain_to_set)

    # 用于gain显示的标签
    def update_gain_label(self):
        # 获取实时曝光
        self.real_gain = self.cam.Gain.GetValue()  # us
        self.lb_gain_show.setText(str(self.real_gain))



    # 用于重置相机gain模式到自动
    def Reset_gain(self):
        if self.cam.Gain.GetAccessMode() != PySpin.RW:
            print('Unable to enable automatic exposure (node retrieval). Non-fatal error...')
        self.cam.GainAuto.SetValue(PySpin.GainAuto_Continuous)
        print('Automatic gain enabled...')
        # self.gain_slider.setValue(self.cam.Gain.GetValue())


    #检查手动输入的曝光值是否是数字 是否在范围内,如果为空不判断
    def check_edit_exp(self):
        if is_number(self.exp_edit.text()) and float(self.exp_edit.text())<self.cam.ExposureTime.GetMax() and float(self.exp_edit.text())>self.cam.ExposureTime.GetMin():
            pass
        elif self.exp_edit.text()=='':
            pass
        else:
            QMessageBox.information(self, "notice", 'Please check if you have entered the correct exposure value')
        print(self.cam.ExposureTime.GetMax(),self.cam.ExposureTime.GetMin())

    # 检查手动输入的增益值是否是数字 是否在范围内,如果为空不判断
    def check_edit_gain(self):
        if is_number(self.gain_edit.text()) and float(self.gain_edit.text())<self.cam.Gain.GetMax() and float(self.gain_edit.text())>self.cam.Gain.GetMin():
            pass
        elif self.gain_edit.text()=='':
            pass
        else:
            QMessageBox.information(self, "notice", 'Please check if you have entered the correct exposure value')
        print(self.cam.Gain.GetMax(),self.cam.Gain.GetMin())

    # 用于获取相机fps
    def update_fps_label(self):
        fps_num=round(self.real_fps, 1)
        self.ui.lb_show_fps.setText(str(fps_num))

    #用于获取保存路径,并保存在类对象里,返回最终保存路径
    def get_save_path(self):
        self.pic_save_path=QFileDialog.getExistingDirectory(self, "choose save path")

        return self.pic_save_path

    # 用于图像开始保存
    def start_save_picture(self):
        self.get_save_path()# return self.pic_save_path

        text, ok = QInputDialog.getInt(self, "Save interval", "ms:")
        if ok:
            self.save_interval=int(text)
            self.img_save_timer.start(self.save_interval)#启动定时器用于更改保存flag
            self.ui.btn_stop_save.setEnabled(True)
            self.ui.label_show_save_inf.setText('Working')
        else:
            QMessageBox.information(self, "notice", 'Please check if you have entered the correct save interval')


    #用于获取一张图像并且保存下来
    def save_one_img(self):
        self.need_save_fig=True

    # 用于停止保存图像
    def stop_save_picture(self):
        #停止后说明保存结束，下次也从img1开始保存
        global flag_save_number
        flag_save_number=1

        #关闭定时器，关闭保存
        self.img_save_timer.stop()
        self.need_save_fig=False
        QMessageBox.information(self, "notice", 'All pictures have been saved in {}'.format(self.pic_save_path))
        self.ui.btn_stop_save.setEnabled(False)
        self.ui.label_show_save_inf.setText('Not Working')
        #flag表示相机已经关闭
        global flag_cam_is_on
        flag_cam_is_on = False
    # 用于创建透镜子类，控制透镜
    def create_len(self):
        try:
            self.lens = Lens(self.ui.comboBox_port.currentText(), debug=False)

            if self.lens.comNum == 0:
                QMessageBox.information(self,'Notice', u'Connection failed, please enter the correct port number and check if the hardware device is plugged in, or just retry！')

            elif self.lens.comNum == 1:
                #状态显示
                self.ui.lb_show_com.setText('Connected')

                #获取固件版本
                self.ui.lb_firmware_show.setText(str(self.lens.firmware_version))

                # 获取相机序列号
                self.ui.lb_lens_SN_show.setText(str(self.lens.lens_serial))

                #获取温度
                temperature = str(self.lens.get_temperature())
                temperature = float(temperature)
                temperature = round(temperature, 2)
                temperature = str(temperature)
                self.ui.lb_temp_show.setText(temperature)

                #获取最大和最小屈光度
                min_diop, max_diop = self.lens.to_focal_power_mode()
                self.ui.lb_posi_dip_show.setText(str(max_diop))
                self.ui.lb_neg_dip_show.setText(str(min_diop))

                #使能按钮
                self.ui.btn_set_diopter.setEnabled(True)
                self.ui.btn_Reset_diopter.setEnabled(True)
                self.ui.btn_cycle_diopter.setEnabled(True)
                self.ui.btn_stop_cycle_diopter.setEnabled(True)
                if self.ui.video_num.text()=='1':
                    self.ui.btn_autoSearch.setEnabled(True)
                    self.ui.btn_autofocus.setEnabled(True)
                # 获取连接上的端口号
                self.port_checked = self.ui.comboBox_port.currentText()

                self.timer_port.start(1000)

        except:
            QMessageBox.critical(self, 'Notice', 'The com is not available or the driver is not installed')

    # 用于把透镜的屈光度设到输入的某一值
    def set_constant_diopter(self,input):
        if self.lens.comNum == 1:
            if is_number(input):
                min_diop, max_diop = self.lens.to_focal_power_mode()
                # 检查输入是否正确
                if input <= max_diop and input >= min_diop:
                    self.lens.set_diopter(input)
                    self.ui.lb_diop_show_2.setText(str(self.lens.get_diopter()))
                else:
                    QMessageBox.information(self, 'Notice',
                                            u'Please enter the correct diopter!Diopter value should be between {} and {}'.format(
                                                min_diop, max_diop))
            else:
                QMessageBox.information(self, 'Notice',
                                        u'Diopter should be number!')
        else:
            QMessageBox.information(self, 'Notice',
                                    u'You have not connected a lens')

    # 用于把透镜的屈光度设到某一值
    def setdiopter(self):
        if self.lens.comNum == 1:
            input_text=self.ui.diopter_input_edit.text()

            if is_number(input_text):
                input_diopter=float(self.ui.diopter_input_edit.text())
                min_diop, max_diop = self.lens.to_focal_power_mode()
                # 检查输入是否正确
                if input_diopter <= max_diop and input_diopter >= min_diop:
                    self.lens.set_diopter(input_diopter)
                    self.ui.lb_diop_show.setText(str(self.lens.get_diopter()))
                else:
                    QMessageBox.information(self, 'Notice',u'Please enter the correct diopter!Diopter value should be between {} and {}'.format(min_diop,max_diop))

            else:
                QMessageBox.information(self, 'Notice',
                                        u'Diopter should be number!')
        else:
            QMessageBox.information(self, 'Notice',
                                    u'You have not connected a lens')
    # 用于重置透镜屈光度
    def reset_diopter(self):
        if self.lens.comNum == 1:
            self.lens.set_diopter(0)
            self.ui.lb_diop_show.setText('0')
        else:
            QMessageBox.information(self, 'Notice',
                                    u'You have not connected a lens')

    # 用于屈光度变化循环
    def begin_cycle(self):
        #检查是否连接到了透镜
        if self.port_checked!='':
            min_diop, max_diop = self.lens.to_focal_power_mode()
            max_cycle = self.ui.max_diopter_input.text()
            min_cycle = self.ui.min_diopter_input.text()
            step = self.ui.step_diopter_input.text()
            cycle = self.ui.cycle_diopter_input.text()
            max_cycle = float(max_cycle)
            min_cycle = float(min_cycle)
            step = float(step)
            #如果没有输入cycle，默认10s单程，防止崩溃
            if cycle=='':
                cycle=10
            else:
                cycle = float(cycle) #s
            save_check =self.ui.btn_save_pic_zoom.isChecked()
            if save_check == True:
                self.get_save_path()  # 创建文件夹并且设置保存路径到类对象中
                # 计数器清0
                global flag_save_number
                flag_save_number = 1

            #判断输入是否正确
            if max_cycle <= max_diop and min_cycle >= min_diop:
                #flag表示需要进行循环
                self.cycle_flag = True
                self.ui.lb_show_zoom.setText('Zooming')
                # 判断是否同时需要保存图像
                save_check = self.ui.btn_save_pic_zoom.isChecked()  # bool
                # 计算需要跳的次数
                times = (max_cycle - min_cycle) / step +1
                # 计算每次调节需要间隔的时间
                every_time = (cycle * 1000) / times  # 计算出来的时间单位ms
                fore_back=self.ui.btn_save_pic_zoom_2.isChecked()#是否需要来回

                self.cycle_thread = Thread_cycle(save_check,fore_back,times,every_time,max_cycle,min_cycle,step)

                self.cycle_thread.change_diopter_signal.connect(self.set_constant_diopter)
                if save_check == True:
                    self.cycle_thread.save_signal.connect(self.cycly_save_img)

                self.cycle_thread.start()

                self.ui.lb_show_zoom.setText('Finish zooming')

            else:
                QMessageBox.information(self, 'Notice',
                                        u'Please enter the correct diopter!Diopter value should be between {} and {}'.format(
                                            min_diop, max_diop))

        else:
            QMessageBox.information(self, 'Notice',
                                    u'You have not connected a lens')

    # 用于停止屈光度变化循环
    def stop_cycle(self):
        self.cycle_thread.terminate()


    #用于计算cycle图像数量：
    def updade_count_number(self):
        #检查3个屈光度的输入是否有一个为空，如果为空，不必要更新
        if (self.max_diopter_input.text()=='' or self.min_diopter_input.text()=='' or self.step_diopter_input.text()==''):
            pass
        else:
            #如果都不是空的了，就检查是否都是数字
            if any(char.isdigit() for char in self.max_diopter_input.text())and any(char.isdigit() for char in self.min_diopter_input.text()) and any(char.isdigit() for char in self.step_diopter_input.text()):
                number_of_pic=int((float(self.max_diopter_input.text())-float(self.min_diopter_input.text()))//float(self.step_diopter_input.text()))
                self.ui.lb_pic_number_show.setText(str(number_of_pic))
            else:
                QMessageBox.information(self, 'Notice',
                                            u'Please check if you have entered the correct diopter and step!')



    # 用于安全退出，释放摄像头
    def closeEvent(self, event):
        self.cam.DeInit()
        del self.cam

        # Clear camera list before releasing system
        self.cam_list.Clear()

        # Release system instance
        self.system.ReleaseInstance()

        event.accept()

    # 用于自动调焦
    def auto_focus(self):
        #检查透镜和相机连接是否已经连接
        # try:
        #获取最大和最小屈光度
        min_diop, max_diop = self.lens.to_focal_power_mode()
        min_diop=float(min_diop)
        max_diop=float(max_diop)
        tol=0.1



        self.ui.lb_show_focus_inf.setText('Focusing')
        if self.ui.comboBox_focus_method.currentText() =='Trichotomy':
            self.focus_thread=ternary_search_thread(min_diop, max_diop, tol)
            self.focus_thread.callback.connect(self.set_and_cal_two_image)
            self.focus_thread.result_signal.connect(self.get_best_diop)
            self.focus_thread.finished.connect(self.autofocus_on_finished)
            self.focus_thread.start()

        elif self.ui.comboBox_focus_method.currentText()=='Climbing':
            self.focus_thread = hill_climb_thread(min_diop, max_diop,step=0.1, max_iter=1000)
            self.focus_thread.callback.connect(self.set_and_cal_one_image)
            self.focus_thread.result_signal.connect(self.get_best_diop)
            self.focus_thread.finished.connect(self.autofocus_on_finished)
            self.focus_thread.start()

        elif self.ui.comboBox_focus_method.currentText()=='Curve fitting':
            self.focus_thread=curvefitting_thread(min_diop,max_diop)
            self.focus_thread.callback.connect(self.set_and_cal_one_image)
            self.focus_thread.result_signal.connect(self.get_best_diop)
            self.focus_thread.finished.connect(self.autofocus_on_finished)
            self.focus_thread.start()

        # except:
        #     QMessageBox.information(self, 'Notice',
        #                             u'Please connect the camera, lens!')
        # finally:
        #     roi_flag == False
            # 定义一个函数fun(x)，这里假设它是单峰的

    #收到变焦结束信号后，设置到最佳屈光度
    def autofocus_on_finished(self):
        self.lens.set_diopter(self.best_diop)
        self.ui.lb_show_focus_inf.setText('Finish')
        self.ui.lb_diop_show.setText(str(round(self.best_diop,2)))

    #设置到diop屈光度并且计算清晰度
    def set_and_cal_one_image(self, diop):
        if self.lens.comNum == 1:
            # 设置屈光度
            self.lens.set_diopter(diop)
            # 获取当前屈光度
            current_diop = self.lens.get_diopter()
            # 设置的值和获取的值误差小于tol,说明设置成功，再计算清晰度
            if abs(current_diop - diop) <= 0.01:
                # 非ROI
                global roi_flag
                if roi_flag == False:
                    out = cal_clarity(self.update_image(), measure=self.ui.comboBox_focus_measure.currentText(),
                                      roi_point=None, label_height=None, roi_height=None, roi_width=None)
                # 局部自动对焦
                else:
                    # 用于计算实际像素画面与label的比例
                    self.label_height = self.ui.video_label.height()
                    out = cal_clarity(self.update_image(), measure=self.ui.comboBox_focus_measure.currentText(),
                                      roi_point=self.roi_point, label_height=self.label_height,
                                      roi_height=self.roi_height, roi_width=self.roi_width)
            # 设置没有成功,加大时间延迟给透镜
            else:
                print('Diopter setting failure')
                self.lens.set_diopter(diop)
                time.sleep(0.2)

        else:
            QMessageBox.information(self, 'Notice',
                                    u'You have not connected a lens')
        # 发射数据给曲线拟合的线程
        if self.ui.comboBox_focus_method.currentText()=='Curve fitting':
            self.focus_thread.data_signal1.emit(out)
        # 发射数据给爬山算法的线程
        if self.ui.comboBox_focus_method.currentText() == 'Climbing':
            self.focus_thread.data_signal1.emit(out)
        return out

    # 设置两个并发送两个结果
    def set_and_cal_two_image(self,data):
        diop1,diop2=data
        self.focus_result1 = self.set_and_cal_one_image(diop1)
        self.focus_result2 = self.set_and_cal_one_image(diop2)
        self.focus_thread.data_signal1.emit(self.focus_result1)
        self.focus_thread.data_signal2.emit(self.focus_result2)
    def get_best_diop(self,data):
        self.best_diop=data
    #开始录像
    def start_recording(self):
        if self.cam_list.GetSize()!=0:

            try:
                self.video_save_path = QFileDialog.getExistingDirectory(self, "choose save path")
                self.recorder = PySpin.SpinVideo()
                if self.ui.comboBox_output_format_3.currentText()=='AVI':
                    self.option = PySpin.AVIOption()
                    self.option.frameRate =self.real_fps

                elif self.ui.comboBox_output_format_3.currentText()=='MPEG':
                    self.option = PySpin.MJPGOption()
                    self.option.frameRate = self.real_fps
                    text, ok = QInputDialog.getInt(self, "Image quality", "Percentage (0-100):")
                    if ok:
                        self.option.quality = int(text)
                self.recorder.Open(self.video_save_path + '/' + time.strftime('%Y-%m-%d_%H_%M_%S',
                                                                                  time.localtime(
                                                                                      time.time())),self.option)
                self.record_flag=True
                self.ui.btn_stop_save_2.setEnabled(True)

                self.ui.label_show_save_inf.setText('Working')
            except:
                QMessageBox.information(self, 'Notice',
                                        u'Something wrong!')

        else:
            QMessageBox.information(self, 'Notice',
                                    u'Please connect the camera!')


    # 结束录像
    def stop_recording(self):
        if self.record_flag == True:
            try:
                for i in range(len(self.frameslist)):
                    self.recorder.Append(self.frameslist[i])
                self.recorder.Close()

                del self.frameslist
                self.record_flag = False

                QMessageBox.information(self, 'Notice',
                                        u'Recording completed, video stored at {}'.format(self.video_save_path))
                self.ui.btn_stop_save_2.setEnabled(False)
                self.ui.label_show_save_inf.setText('Not Working')
            except:
                QMessageBox.information(self, 'Notice',
                                        u'Failed to save video!')

        else:
            QMessageBox.information(self, 'Notice',
                                    u'You have not started recording yet!')

    # 局部对焦
    def roi_focus(self, event):
        # 获取点击位置
        global roi_flag
        x = event.x()
        y = event.y()
        roi = (x, y)
        try:

            roi_flag = True
            self.roi_point = roi
            self.auto_focus()
        except:
            QMessageBox.information(self, 'Notice',
                                    u'Focusing failure')

    #设置roi_radius
    def set_focus(self):
        max_num = round(min(self.ui.video_label.width(),self.ui.video_label.height())/2)
        text, ok = QInputDialog.getInt(self, "Set ROI Radius", "Radius(0-{})  default=50".format(str(max_num)))
        if ok:
            self.roi_height = int(text)
            self.roi_width = int(text)
        else:
            QMessageBox.information(self, "notice", 'Please check if you have entered the correct Radius, default roi radius=50')

    def boxout_mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.move_flag = False

    def boxout_mouseMoveEvent(self, event):
        if not self.start_pos:
            return
        self.move_flag=True
        self.end_pos = event.pos()
        self.update()

    def boxout_mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.end_pos = event.pos()
            self.update()

            if self.move_flag==True:
                self.boxout_focus()
            else:
                pass

    def boxout_focus(self):
        print(self.start_pos, self.end_pos)
        x1 = self.start_pos.x()
        y1 = self.start_pos.y()
        x2 = self.end_pos.x()
        y2 = self.end_pos.y()
        self.roi_point = (round(x1 + x2) / 2, round(y1 + y2) / 2)
        self.roi_height = abs(round(y2 - y1))
        self.roi_width = abs(round(x2 - x1))
        global roi_flag
        roi_flag = True
        self.auto_focus()
        self.start_pos = None
        self.end_pos = None
        self.drawRectFlag = False
        self.draw_rect = QRect()
        self.update()

    #用于自动找到聚焦的上下平面
    def auto_search(self):
        # 检查是否连接到了透镜
        if self.port_checked!='':
            try:
                self.diop_list=[]
                min_diop, max_diop = self.lens.to_focal_power_mode()
                max_cycle = max_diop
                min_cycle = min_diop
                step = 0.1#步距
                cycle = float(10.0)#设定总时间
                max_cycle = float(max_cycle)
                min_cycle = float(min_cycle)
                step = float(step)#设定步距
                # 计算需要跳的次数
                times = (max_cycle - min_cycle) / step +1
                # 计算每次调节需要间隔的时间
                every_time = (cycle * 1000) / times  # 计算出来的时间单位ms

                self.ui.lb_show_zoom.setText('Searching')

                #开启变焦子线程
                self.search_thread=Thread_search(times,every_time,max_cycle,min_cycle,step)
                #绑定变焦函数
                self.search_thread.change_diopter_signal.connect(self.set_constant_diopter)
                #绑定添加图片到列表的函数
                self.search_thread.add_frame_signal.connect(self.add_frame2list)
                #绑定label显示的函数
                self.search_thread.search_state_signal.connect(self.lable2finish)
                #绑定接受屈光度列表的函数
                self.search_thread.add_diopter_list_signal.connect(self.add_diop_list)
                #绑定变焦完后 开始计算清晰度的函数
                self.search_thread.start_to_cal_clarity_signal.connect(self.cal_frame_list_clarity)

                self.search_thread.start()


            except Exception as e:
                print(e)
                QMessageBox.information(self, 'Notice',
                                        u'You have not enter correct diopter and step!')

        else:
            QMessageBox.information(self, 'Notice',
                                    u'You have not connected a lens')
    #变焦完成,计算列表清晰度
    def cal_frame_list_clarity(self):
        print(self.frame_list_cal)
        print(self.diop_list)
    #得到屈光度列表
    def add_diop_list(self,diop):
        self.diop_list.append(diop)

    #自动寻找焦平面的标签
    def lable2finish(self,flag):
        if flag==False:
            self.ui.lb_show_zoom.setText('Finish Searching')
        else:
            self.ui.lb_show_zoom.setText('Zooming')
    #用于融合图像栈的按钮
    def fusion_stack(self):

        stack_path = self.ui.edit_fusion_path.text()
        result_path = self.ui.edit_fusion_result_path.text()
        is_valid_scale=is_scale_valid(self.ui.edit_scale.text())
        use_gpu_check_box=self.ui.btn_use_GPU.isChecked()

        #判断文件夹中的图片是否都是数字名字
        if is_valid_scale:
            # 判断是否都是正确的文件夹路径
            if os.path.isdir(stack_path) and os.path.isdir(result_path):
                try:
                    img_formats = {'.jpg', '.png', '.bmp'}
                    existing_formats = set()
                    #检索存在的图片格式，#需要整个文件夹的图片都是统一格式的图片
                    for root, dirs, files in os.walk(stack_path):
                        for file in files:
                            ext = os.path.splitext(file)[1].lower()
                            existing_formats.add(ext)
                    if len(existing_formats)==1:
                        if existing_formats & img_formats:
                            #判断图片名字是否全是数字
                            for filename in os.listdir(stack_path):
                                is_number_named=True
                                if not re.match(r'^\d+.jpg$', filename):
                                    is_number_named = False
                                else:
                                    pass
                            if is_number_named==True:
                                [source_format] = existing_formats
                                #如果真的可以用gpu,就用gpu
                                if gpu_is_available()&use_gpu_check_box:
                                    use_gpu_in_fact=True
                                #如果不可以用
                                else:
                                    use_gpu_in_fact=False
                                    if use_gpu_check_box==True:
                                        QMessageBox.information(self, 'Notice',
                                                                u'No GPU found, use CPU instead.')
                                    else:
                                        pass

                                fusion_thread=fusion.Fusion_stack(stack_path,result_path=result_path,
                                                    source_format=source_format,
                                                    save_format=self.ui.comboBox_output_format_4.currentText(),
                                                    Using_Optimised_Processing=self.ui.btn_use_filter.isChecked(),
                                                    image_scale=0.01*int(self.ui.edit_scale.text()),
                                                    use_gpu=use_gpu_in_fact)


                                QMessageBox.information(self, 'Notice',
                                                        u'{}'.format(fusion_thread.result_path_and_time))


                            else:
                                QMessageBox.information(self, 'Notice',
                                                        u'Images need to be named in numerical order, e.g. 1.jpg, 2.jpg...')
                        else:
                            QMessageBox.information(self, 'Notice',
                                                    u'Only support jpg,png,bmp image format')
                    else:
                        QMessageBox.information(self, 'Notice',
                                                u'There are multiple formats of images in the folder, currently only supports all images in the same format')


                except Exception as e:
                    QMessageBox.information(self, 'Notice',
                                            u'Fusion failed, {}'.format(str(e)))
            else:
                QMessageBox.information(self, 'Notice',
                                        u'You have not enter a correct path')
        else:
            QMessageBox.information(self, 'Notice',
                                    u'Scale needs to be between 1-100')

    # 用于选取融合图像栈的路径
    def browse_stack_dir(self):
        img_stack_path=QFileDialog.getExistingDirectory(self, "Please select the folder path where the image stack is located")

        if img_stack_path:
            self.ui.edit_fusion_path.setText(img_stack_path)

        else:
            QMessageBox.information(self, 'Notice',
                                    u'You have not selected path')


    # 用于选取融合图像栈保存的路径
    def browse_fusion_result_dir(self):
        fusion_result_save_path=QFileDialog.getExistingDirectory(self, "Please select the folder path where the image stack is located")
        if fusion_result_save_path:
            self.ui.edit_fusion_result_path.setText(fusion_result_save_path)
        else:
            QMessageBox.information(self, 'Notice',
                                    u'You have not selected path')

    # 变焦循环过程中的保存flag，flag来自于信号
    def cycly_save_img(self, flag):
        self.need_save_fig = flag

    #寻找焦平面过程中的保存flag，flag来自于信号
    def add_frame2list(self, flag):
        self.cal_auto_search=flag
#子线程用来保存图片，不影响主线程显示
class Save_img(QThread):
    def __init__(self,data,path,save_format):
        self.data=data
        self.path=path
        self.save_format='.'+save_format
        self.save()
    # 重写run方法
    def save(self):
        global flag_save_number#定义在文件头
        self.data.Save(os.path.join(self.path,str(flag_save_number)+ self.save_format))
        print('{}{} has been saved in {}'.format(flag_save_number,self.save_format,self.path + '/'+str(flag_save_number)+'%s'%self.save_format))
        flag_save_number += 1#增量式创建，图片按先后排序

class Thread_cycle(QThread):
    # 创建信号，发送str类型数据
    save_signal = pyqtSignal(bool)
    change_diopter_signal=pyqtSignal(float)
    def __init__(self,save_check,check_front_back,times,every_time,max_cycle,min_cycle,step):
        super().__init__()
        self.save_check=save_check
        self.times=times
        self.every_time=every_time
        self.max_cycle=max_cycle
        self.min_cycle=min_cycle
        self.step=step
        self.check_front_back=check_front_back

    def run(self):
        # 从最大开始拍到最小
        if not self.check_front_back:
            self.real_diop = self.max_cycle

            for i in range(int(self.times)):
                self.change_diopter_signal.emit(self.real_diop)
                # 等待变焦
                time.sleep(0.2)
                self.save_signal.emit(self.save_check)
                # 等待保存
                time.sleep(0.2)
                self.real_diop -= self.step
                time.sleep(self.every_time / 1000)  # time.sleep(s)
                # 防卡
                QApplication.processEvents()

        # 来回拍
        else:
            # 先从最大开始拍到最小
            self.real_diop = self.max_cycle
            for i in range(int(self.times)):
                self.change_diopter_signal.emit(str(self.real_diop))

                self.save_signal.emit(self.save_check)

                self.real_diop -= self.step
                time.sleep(self.every_time / 1000)  # time.sleep(s)

                QApplication.processEvents()

            # 再从最小开始拍到最大,这里不重复拍所以先加上step
            self.real_diop = self.min_cycle + self.step
            for i in range(int(self.times)):

                self.change_diopter_signal.emit(str(self.real_diop))
                self.save_signal.emit(self.save_check)

                self.real_diop += self.step
                time.sleep(self.every_time / 1000)  # time.sleep(s)
                QApplication.processEvents()

class Thread_search(QThread):
    # 创建信号，发送str类型数据
    change_diopter_signal=pyqtSignal(float)
    add_frame_signal=pyqtSignal(bool)
    search_state_signal=pyqtSignal(bool)
    start_to_cal_clarity_signal=pyqtSignal(bool)
    add_diopter_list_signal=pyqtSignal(float)

    def __init__(self,times,every_time,max_cycle,min_cycle,step):
        super().__init__()
        self.times=times
        self.every_time=every_time
        self.max_cycle=max_cycle
        self.min_cycle=min_cycle
        self.step=step

    def run(self):
        # 从最大开始拍到最小
        self.real_diop = self.max_cycle
        self.search_state_signal.emit(True)
        self.start_to_cal_clarity_signal.emit(False)

        for i in range(int(self.times)):
            self.change_diopter_signal.emit(self.real_diop)
            self.add_diopter_list_signal.emit(self.real_diop)
            # 等待变焦
            time.sleep(0.2)

            self.add_frame_signal.emit(True)
            # 等待保存
            time.sleep(0.2)

            self.real_diop -= self.step
            time.sleep(self.every_time / 1000)  # time.sleep(s)
            QApplication.processEvents()
        self.search_state_signal.emit(False)
        self.start_to_cal_clarity_signal.emit(True)


class ternary_search_thread(QThread):
    callback = pyqtSignal(tuple)
    data_signal1 = pyqtSignal(float)
    data_signal2 = pyqtSignal(float)
    result_signal = pyqtSignal(float)
    def __init__(self,min_diop, max_diop, tol):
        super().__init__()
        self.min_diop=min_diop
        self.max_diop=max_diop
        self.tol=tol

    def run(self):
        self.data_signal1.connect(self.on_signal1)
        self.data_signal2.connect(self.on_signal2)

        # 定义一个函数fun(x)，这里假设它是单峰的
        def ternary_search(a, b, eps):
            # 当区间长度小于eps时，停止循环
            while b - a > eps:
                # 计算两个分割点

                m1 = a + (b - a) / 3
                m2 = b - (b - a) / 3
                # 比较函数值，判断最大值在哪个区间内
                send=(m1,m2)
                self.callback.emit(send)
                time.sleep(1)#等待计算一秒，配置差可能要增大不然会报错
                fun1_result=self.fun1_result
                fun2_result=self.fun2_result
                if fun1_result < fun2_result:
                    # 最大值在[m1, b]内，舍弃[a, m1]区间
                    a = m1
                else:
                    # 最大值在[a, m2]内，舍弃[m2, b]区间
                    b = m2
            # 返回区间中点作为最大值的近似解
            return (a + b) / 2

        self.result=ternary_search(self.min_diop,self.max_diop,self.tol)
        self.result_signal.emit(self.result)


    def on_signal1(self, float):
        self.fun1_result=float

    def on_signal2(self, float):
        self.fun2_result=float

#曲线拟合自动对焦子线程
class curvefitting_thread(QThread):
    callback = pyqtSignal(float)
    data_signal1 = pyqtSignal(float)

    result_signal = pyqtSignal(float)

    def __init__(self, min_diop, max_diop):
        super().__init__()
        self.min_diop = min_diop
        self.max_diop = max_diop


    def run(self):
        self.data_signal1.connect(self.on_signal1)

        def curvefitting(min,max):
            # x和y数据
            step=0.1
            x = np.arange(min,max,step)
            x_list=x.tolist()
            #随机生成y，用于更新
            y = np.arange(min, max, step)
            #计算y
            for val_index,val in enumerate(x_list):

                self.callback.emit(val)
                time.sleep(0.5)#等待计算一秒，配置差可能要增大不然会报错
                fun1_result = self.fun1_result
                y[val_index]=fun1_result
            # 需要拟合的二次函数
            def func(x, a, b, c):
                return a*x*x+b*x+c
            popt, pcov = curve_fit(func, x, y)

            a,b,c=popt
            #求y最大值时候的x
            x=max_y(a,b,c,self.min_diop,self.max_diop)
            return x

        self.result = curvefitting(self.min_diop, self.max_diop)
        self.result_signal.emit(self.result)

    def on_signal1(self, float):
        self.fun1_result = float

#爬山子线程
class hill_climb_thread(QThread):
    callback = pyqtSignal(float)
    data_signal1 = pyqtSignal(float)
    result_signal = pyqtSignal(float)

    def __init__(self, min_diop, max_diop,step, max_iter):
        super().__init__()
        self.min_diop = min_diop
        self.max_diop = max_diop
        self.step=step
        self.max_iter=max_iter

    def run(self):
        self.data_signal1.connect(self.on_signal1)

        def hill_climb(min, max,step, max_iter):
            x = np.random.uniform(min, max)
            print(x)
            for i in range(max_iter):
                self.callback.emit(x)
                time.sleep(0.5)  # 等待计算一秒，配置差可能要增大不然会报错
                curr_val = self.fun1_result

                next_x = x + np.random.uniform(-step, step)
                self.callback.emit(next_x)
                time.sleep(0.5)  # 等待计算一秒，配置差可能要增大不然会报错
                next_val = self.fun1_result
                print(next_val,curr_val)
                if next_val > curr_val:
                    x, curr_val = next_x, next_val
            return x


        self.result = hill_climb(self.min_diop, self.max_diop,self.step, self.max_iter)
        self.result_signal.emit(self.result)

    def on_signal1(self, float):
        self.fun1_result = float