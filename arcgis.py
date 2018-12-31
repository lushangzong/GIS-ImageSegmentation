####################################################
#作者：鲁尚宗 时间:2018年12月31日
#图像分割中分水岭算法的探索
#通过GIS方法实现图像分割
#主要使用了提取山脊线的方法和等高线的方法
#####################################################
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


# ##################################################
# 绘制图像数组的函数，传入一个图像数组，绘制它的图像
# array为实数数组，name为图片的名字
# ##################################################
def draw(array, name):
    plt.imshow(array, plt.cm.gray)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title(name, fontproperties='SimHei')  # 原始图像 图像题目
    plt.show()


# ##################################################
# 细化时水平扫描函数
# 输入要细化的图像和参考矩阵
# 会改变图像矩阵
# 采用查表法判断像素是否可以删除
# ##################################################
def VThin(image, array):
    h, w = image.shape
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i, j - 1] + image[i, j] + image[i, j + 1] if 0 < j < w - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 255:
                                a[k * 3 + l] = 1
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i, j] = array[sum] * 255
                    if array[sum] == 1:
                        NEXT = 0
    return image


# ##################################################
# 细化时垂直扫描函数
# 输入要细化的图像和参考矩阵
# 会改变图像矩阵
# 采用查表法判断像素是否可以删除
# ##################################################
def HThin(image, array):
    h, w = image.shape
    NEXT = 1
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i - 1, j] + image[i, j] + image[i + 1, j] if 0 < i < h - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 255:
                                a[k * 3 + l] = 1
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i, j] = array[sum] * 255
                    if array[sum] == 1:
                        NEXT = 0
    return image


# ##################################################
# 细化函数
# 采用查表法判断像素是否可以删除
# 输入要细化的图像和参考矩阵，默认迭代次数为10次
# 在每行水平扫描的过程中，先判断每一点的左右邻居，如果都是黑点，则该点不做处理。
# 另外，如果某个黑店被删除了，
# 则跳过它的右邻居，处理下一点。对矩形这样做完一遍，水平方向会减少两像素。
# 然后我们再改垂直方向扫描，方法一样。
# 这样做一次水平扫描和垂直扫描，原图会“瘦”一圈
# 多次重复上面的步骤，知道图形不在变化为止
# ##################################################
def Xihua(image, array, num=10):
    iXihua = image.copy()
    iXihua = np.uint16(iXihua)

    # 循环扫描直到结果不再变化为止
    for i in range(num):
        VThin(iXihua, array)
        HThin(iXihua, array)

    iXihua = np.uint8(iXihua)
    return iXihua


# 读入灰度图片
filename = 'test.gif'
im_PIL = Image.open(filename)
img = np.array(im_PIL)

draw(img, '原图')
# 获取大小
height, width = img.shape

# 储存坡度
slope = np.zeros([height - 2, width - 2], np.float64)

# 计算坡度
for i in range(height - 2):
    for j in range(width - 2):
        dx = ((img[i, j + 2] + 2 * img[i + 1, j + 2] + img[i + 2, j + 2]) - (
                img[i, j] + 2 * img[i + 1, j] + img[i + 2, j])) / 8
        dy = ((img[i + 2, j] + 2 * img[i + 2, j + 1] + img[i + 2, j + 2]) - (
                img[i, j] + 2 * img[i, j + 1] + img[i, j + 2])) / 8
        slope[i, j] = np.sqrt(dx ** 2 + dy ** 2)

draw(slope, '坡度')
# 洼地填充,填充比较浅的区域，防止过于碎片化

Threshold = np.mean(slope)
for i in range(height - 2):
    for j in range(width - 2):
        # 小于阈值的全部填充
        if slope[i, j] < Threshold:
            slope[i, j] = Threshold

draw(slope, '填充后')

# 用来储存流向
direction = np.zeros([height - 2, width - 2], np.uint8)

# 先确定最外一圈的流向
# 最左边的一列 列序号为0
for i in range(height - 2):
    # 第一个点
    if i == 0:
        # 相邻三个的最小值
        min = np.min([slope[i, 1], slope[i + 1, 0], slope[i + 1, 1]])
        if min == slope[i, 1]:  # 往右边流
            direction[i, 0] = 1
        elif min == slope[i + 1, 0]:  # 往下边流
            direction[i, 0] = 4
        elif min == slope[i + 1, 1]:  # 往右下流
            direction[i, 0] = 2

    # 最后一个点
    elif i == height - 3:
        # 相邻三个的最小值
        min = np.min([slope[i, 1], slope[i - 1, 0], slope[i - 1, 1]])
        if min == slope[i, 1]:  # 往往右边流
            direction[i, 0] = 1
        elif min == slope[i - 1, 0]:  # 往上边流
            direction[i, 0] = 64
        elif min == slope[i - 1, 1]:  # 往右上流
            direction[i, 0] = 128

    # 中间的点
    else:
        # 相邻五个的最小值
        min = np.min([slope[i, 1], slope[i - 1, 0], slope[i - 1, 1], slope[i + 1, 0], slope[i + 1, 1]])
        if min == slope[i, 1]:  # 往右边流
            direction[i, 0] = 1
        elif min == slope[i + 1, 0]:  # 往下边流
            direction[i, 0] = 4
        elif min == slope[i + 1, 1]:  # 往右下流
            direction[i, 0] = 2
        elif min == slope[i - 1, 0]:  # 往上边流
            direction[i, 0] = 64
        elif min == slope[i - 1, 1]:  # 往右上流
            direction[i, 0] = 128

# 最右边的一列，列序号为width - 3
for i in range(height - 2):
    # 第一个点
    if i == 0:
        # 相邻的三个点的最小值
        min = np.min([slope[i, width - 4], slope[i + 1, width - 3], slope[i + 1, width - 4]])
        if min == slope[i, width - 4]:  # 往左流
            direction[i, width - 3] = 16
        elif min == slope[i + 1, width - 3]:  # 往下流
            direction[i, width - 3] = 4
        elif min == slope[i + 1, width - 4]:  # 往左下流
            direction[i, width - 3] = 8

        # 最后一个点
    elif i == height - 3:
        # 相邻三个点的最小值
        min = np.min([slope[i, width - 4], slope[i - 1, width - 3], slope[i - 1, width - 4]])
        if min == slope[i, width - 4]:  # 往左流
            direction[i, width - 3] = 16
        elif min == slope[i - 1, width - 3]:  # 往上流
            direction[i, width - 3] = 64
        elif min == slope[i - 1, width - 4]:  # 往左上流
            direction[i, width - 3] = 32

    # 中间的点
    else:
        # 相邻五个点的最小值
        min = np.min([slope[i, width - 4], slope[i + 1, width - 3], slope[i + 1, width - 4], slope[i - 1, width - 3],
                      slope[i - 1, width - 4]])
        if min == slope[i, width - 4]:  # 往左流
            direction[i, width - 3] = 16
        elif min == slope[i + 1, width - 3]:  # 往下流
            direction[i, width - 3] = 4
        elif min == slope[i + 1, width - 4]:  # 往左下流
            direction[i, width - 3] = 8
        elif min == slope[i - 1, width - 3]:  # 往上流
            direction[i, width - 3] = 64
        elif min == slope[i - 1, width - 4]:  # 往左上流
            direction[i, width - 3] = 32

# 最上边的一行，行序号为0
for j in range(width - 2):
    # 第一个点
    if j == 0:
        # 相邻三个的最小值
        min = np.min([slope[1, j], slope[0, j + 1], slope[1, j + 1]])
        if min == slope[1, j]:  # 往下流
            direction[0, j] = 4
        elif min == slope[0, j + 1]:  # 往右流
            direction[0, j] = 1
        elif min == slope[1, j + 1]:  # 往右下流
            direction[0, j] = 2

    # 最后一个点
    elif j == width - 3:
        # 相邻三个的最小值
        min = np.min([slope[1, j], slope[0, j - 1], slope[1, j - 1]])
        if min == slope[1, j]:  # 往下流
            direction[0, j] = 4
        elif min == slope[0, j - 1]:  # 往左流
            direction[0, j] = 16
        elif min == slope[1, j - 1]:  # 往左下流
            direction[0, j] = 8
    # 中间的点
    else:
        # 相邻五个的最小值
        min = np.min([slope[1, j], slope[0, j + 1], slope[1, j + 1], slope[0, j - 1], slope[1, j - 1]])
        if min == slope[1, j]:  # 往下流
            direction[0, j] = 4
        elif min == slope[0, j + 1]:  # 往右流
            direction[0, j] = 1
        elif min == slope[1, j + 1]:  # 往右下流
            direction[0, j] = 2
        elif min == slope[0, j - 1]:  # 往左流
            direction[0, j] = 16
        elif min == slope[1, j - 1]:  # 往左下流
            direction[0, j] = 8

# 最下边的一行，行序号为height - 3
for j in range(width - 2):
    # 第一个点
    if j == 0:
        # 相邻三个的最小值
        min = np.min([slope[height - 4, j], slope[height - 3, j + 1], slope[height - 4, j + 1]])
        if min == slope[height - 4, j]:  # 往上流
            direction[height - 3, j] = 64
        elif min == slope[height - 3, j + 1]:  # 往右流
            direction[height - 3, j] = 1
        elif min == slope[height - 4, j + 1]:  # 往右上流
            direction[height - 3, j] = 128
        # 最后一个点
    elif j == width - 3:
        # 相邻三个的最小值
        min = np.min([slope[height - 4, j], slope[height - 3, j - 1], slope[height - 4, j - 1]])
        if min == slope[height - 4, j]:  # 往上流
            direction[height - 3, j] = 64
        elif min == slope[height - 3, j - 1]:  # 往左流
            direction[height - 3, j] = 16
        elif min == slope[height - 4, j - 1]:  # 往左上流
            direction[height - 3, j] = 32
    # 中间的点
    else:
        # 相邻五个的最小值
        min = np.min(
            [slope[height - 4, j], slope[height - 3, j + 1], slope[height - 4, j + 1], slope[height - 3, j - 1],
             slope[height - 4, j - 1]])
        if min == slope[height - 4, j]:  # 往上流
            direction[height - 3, j] = 64
        elif min == slope[height - 3, j + 1]:  # 往右流
            direction[height - 3, j] = 1
        elif min == slope[height - 4, j + 1]:  # 往右上流
            direction[height - 3, j] = 128
        elif min == slope[height - 3, j - 1]:  # 往左流
            direction[height - 3, j] = 16
        elif min == slope[height - 4, j - 1]:  # 往左上流
            direction[height - 3, j] = 32

# 除了边缘外中间的点
for i in range(1, height - 3):
    for j in range(1, width - 3):
        # 相邻八个点的最小值
        min = np.min([slope[i - 1, j - 1], slope[i, j - 1], slope[i + 1, j - 1],
                      slope[i - 1, j], slope[i + 1, j],
                      slope[i - 1, j + 1], slope[i, j + 1], slope[i + 1, j + 1]])
        if min == slope[i - 1, j - 1]:  # 往左上流
            direction[i, j] = 32
        elif min == slope[i, j - 1]:  # 往左流
            direction[i, j] = 16
        elif min == slope[i + 1, j - 1]:  # 往左下流
            direction[i, j] = 8
        elif min == slope[i - 1, j]:  # 往上流
            direction[i, j] = 64
        elif min == slope[i + 1, j]:  # 往下流
            direction[i, j] = 4
        elif min == slope[i - 1, j + 1]:  # 往右上流
            direction[i, j] = 128
        elif min == slope[i, j + 1]:  # 往右流
            direction[i, j] = 1
        elif min == slope[i + 1, j + 1]:  # 往右下流
            direction[i, j] = 2

draw(direction, '方向')

# 用来储存结果
result = np.zeros([height - 2, width - 2], np.uint8)

# 遍历每一个流向，如果一个栅格周围没有水流入则这就是山脊
for i in range(1, height - 3):
    for j in range(1, width - 3):
        # 只要右任何一个方向有水流入，这就不是山脊
        if direction[i - 1, j - 1] == 2 or direction[i - 1, j] == 4 or direction[i - 1, j + 1] == 8 or \
                direction[i, j - 1] == 1 or direction[i, j + 1] == 16 or \
                direction[i + 1, j - 1] == 128 or direction[i + 1, j] == 64 or direction[i + 1, j + 1] == 32:
            result[i, j] = 255
        else:
            result[i, j] = 0

draw(result, '结果')

# 细化时需要查找的表
array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, \
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, \
         1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]

xihua = Xihua(result, array)

draw(xihua, '细化')

# 等高线算法，直接调的函数
x = np.linspace(0, width, width)
y = np.linspace(height, 0, height)
# 将原始数据变成网格数据
X, Y = np.meshgrid(x, y)

# 填充颜色
plt.contourf(X, Y, img, 5, alpha=1, cmap=plt.cm.hot)
plt.contour(X, Y, img, 5, colors='black')

# 绘制等高线
plt.show()
