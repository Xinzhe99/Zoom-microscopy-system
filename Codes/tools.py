import os
import re
from PIL import Image
import cv2
import math
import numpy as np
def config_model_dir(project_dir = os.getcwd(),resume=False,subdir_name='models'):
    # 获取当前项目目录
    # 获取models文件夹的路径
    models_dir = os.path.join(project_dir, subdir_name)
    # 如果models文件夹不存在，就创建它
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    # 如果是第一次运行 创建models0
    if not os.path.exists(os.path.join(models_dir,subdir_name+'1')):
        os.mkdir(os.path.join(models_dir,subdir_name+'1'))
        return os.path.join(models_dir,subdir_name+'1')
    else:
        # 获取models文件夹下已有的子文件夹
        sub_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        sub_dirs.sort(key=lambda l: int(re.findall('\d+', l)[0]))
        last_numbers=re.findall("\d+",sub_dirs[-1])#list
        if resume==False:
            new_sub_dir_name = subdir_name + str(int(last_numbers[0]) + 1)
        else:
            new_sub_dir_name = subdir_name + str(int(last_numbers[0]))
        model_dir_path = os.path.join(models_dir, new_sub_dir_name)
        if resume == False:
            os.mkdir(model_dir_path)
        else:
            pass
        # 把创建的路径存在model_save变量里
        return model_dir_path

def nearest_odd(n):
  if n % 2 == 0: # n是偶数
    return n + 1 # 返回较大的奇数
  else: # n是奇数
    return n # 返回n本身

#获取文件夹中图片的长宽
def get_pic_size_in_dir(folder):
    # 遍历文件夹中的所有文件
    for file in os.listdir(folder):
        # 拼接文件夹路径和文件名
        filename = os.path.join(folder, file)
        # 判断是否是图片文件
        if os.path.isfile(filename) and file.endswith((".jpg", ".png", ".bmp")):
            # 打开图片并获取尺寸
            img = Image.open(filename)
            width, height = img.size
            break
            # 返回第一张图片的宽度和高度
    return width, height


#获取文件夹中图片的类型
def get_first_image_format(folder):
    # 遍历文件夹中的所有文件
    for file in os.listdir(folder):
        # 拼接文件夹路径和文件名
        filename = os.path.join(folder, file)
        # 判断是否是图片文件
        if os.path.isfile(filename) and file.endswith((".jpg", ".png", ".bmp")):
            # 获取图片的扩展名
            ext = os.path.splitext(file)[1]
            break
            # 返回第一张图片的格式
    return ext

#判断文件类型结尾是否以xxx结尾
def judge_format(path):
    # 假设model_path是你要判断的文件路径
    # 获取文件名
    filename = os.path.basename(path)

    # 判断文件名是否以指定的后缀名结尾
    if filename.endswith((".pth", ".pt", ".ckpt")):
        # 如果是，输出"yes"
        return True
    else:
        # 如果不是，输出"no"
        return False

#批量去除文件夹里的空格和括号
import os
def rename_files(path):
    for filename in os.listdir(path):
        new_filename = filename.replace(" ", "").replace("(", "").replace(")", "")
        os.rename(os.path.join(path, filename), os.path.join(path, new_filename))

#判断文本内容是不是数字
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


#判断有没有中文字符
def has_chinese_char(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    return pattern.search(text)

#用于计算图像清晰度
def cal_clarity(frame,measure,label_height,roi_point,roi_height,roi_width):
    # 使用Laplacian算子计算图像的梯度

    if len(frame.shape) == 2:
        image_gray = frame
    else:
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if roi_point==None:
        img=cv2.resize(image_gray,(image_gray.shape[0]//4,image_gray.shape[1]//4))
    else:
        rate_frame2lable=frame.shape[0]/label_height #>1
        (a,b)=roi_point
        x1=round(round(a-roi_width/2)*rate_frame2lable)
        x2=round(round(a+roi_width/2)*rate_frame2lable)
        y1 = round(round(b - roi_height / 2)*rate_frame2lable)
        y2 = round(round(b + roi_height / 2)*rate_frame2lable)
        # 截取区域
        img_roi_label = image_gray[y1:y2, x1:x2]
        img = cv2.resize(img_roi_label, (img_roi_label.shape[0] // 2, img_roi_label.shape[1] // 2))

    if measure=='Laplacian':
        out = cv2.Laplacian(img, cv2.CV_64F).var()
    elif measure == 'Brenner':

        out = 0
        for x in range(0, img.shape[0] - 2):
            for y in range(0, img.shape[1]):
                out += (int(img[x + 2, y]) - int(img[x, y])) ** 2

    elif measure == 'SMD':

        out = 0
        for x in range(0, img.shape[0] - 1):
            for y in range(1, img.shape[1]):
                out += math.fabs(int(img[x, y]) - int(img[x, y - 1]))
            out += math.fabs(int(img[x, y] - int(img[x + 1, y])))

    elif measure== 'SMD2':

        out = 0
        for x in range(0, img.shape[0] - 1):
            for y in range(0, img.shape[1] - 1):
                out += math.fabs(int(img[x, y]) - int(img[x + 1, y])) * math.fabs(
                    int(img[x, y] - int(img[x, y + 1])))

    elif measure== 'Variance':
        out = 0
        u = np.mean(img)

        for x in range(0, img.shape[0]):
            for y in range(0, img.shape[1]):
                out += (img[x, y] - u) ** 2

    elif measure== 'Vollath':

        u = np.mean(img)
        out = -img.shape[0] * img.shape[1] * (u ** 2)
        for x in range(0, img.shape[0] - 1):
            for y in range(0, img.shape[1]):
                out += int(img[x, y]) * int(img[x + 1, y])

    elif measure== 'Energy':
        out = 0
        for x in range(img.shape[0] - 1):
            for y in range(img.shape[1] - 1):
                out += ((int(img[x + 1, y]) - int(img[x, y])) ** 2) + ((int(img[x, y + 1] - int(img[x, y]))) ** 2)
    elif measure == 'Entropy':
        out = 0
        # 计算图片的宽度和高度
        width, height = img.shape

        # 计算图片的像素总数
        total_pixels = width * height

        # 计算每个灰度值出现的次数
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # 返回一个256维的数组，表示每个灰度值出现的次数

        # 计算每个灰度值出现的概率
        prob = hist / total_pixels  # 对数组进行除法运算，得到每个灰度值出现的概率

        # 计算图片信息熵
        for p in prob:  # 遍历每个概率值
            if p > 0:  # 如果概率大于0，才参与计算，否则跳过
                out += -p * math.log2(p)  # 累加每个概率乘以其对数的结果
        return out
    elif measure=='Tenengrad':
        out=0
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        out=np.mean(dst)
# 返回图像的方差，作为清晰度的指标
    return out


def quick_search(a,b):
  # 设置一个容忍误差
  epsilon = 0.1
  # 判断范围是否有效
  if a > b:
    print("Invalid range!")
    return None
  # 循环直到范围小于误差
  while b - a > epsilon:
    # 取范围的中点
    mid = (a + b) / 2
    # 计算中点左右两侧的函数值
    left = cal_clarity(mid - epsilon)
    right = cal_clarity(mid + epsilon)
    # 如果左侧大于右侧，说明最大值在左半区间，更新右端点为中点
    if left > right:
      b = mid
    # 如果右侧大于左侧，说明最大值在右半区间，更新左端点为中点
    elif right > left:
      a = mid
    # 如果左右相等，说明中点就是最大值，返回中点
    else:
      return mid
  # 返回范围的中点作为最终结果
  return (a + b) / 2
#寻找二次函数最大值
def max_y(a: float, b: float, c: float, x_min: float, x_max: float) -> float:
    if a > 0:
        x = -b / (2 * a)
        return max(x_min, min(x_max, x))
    elif a == 0:
        if b >= 0:
            return x_min
        else:
            return x_max
    else:
        if b > 0:
            return x_min
        elif b == 0:
            return -b / (2 * a)
        else:
            x = -b / (2 * a)
            if x_min <= x <= x_max:
                return x
            elif x < x_min:
                return x_min
            else:
                return x_max
#数字在1-100之间
def is_scale_valid(image_scale):
  pattern = r"^([1-9][0-9]?|100)$"
  result = re.match(pattern, image_scale)
  if result:
    return True
  else:
    return False
