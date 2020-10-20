# coding=utf-8
import os
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import requests as rq
import json
import copy

class Board:
    def __init__(self, stat, pos, step=0, preboard=None, prepath=""):
        self.stat = stat
        self.pos = pos
        self.step = step
        self.cost = self.cal_cost()
        self.preboard = preboard
        self.prepath = prepath

    def cal_cost(self):
        count = 0
        sheet = [[0, 0], [0, 1], [0, 2],
                 [1, 0], [1, 1], [1, 2],
                 [2, 0], [2, 1], [2, 2]]
        for i in range(9):
            if self.stat[i] < 0:
                continue
            count += abs(sheet[i][0] - sheet[self.stat[i]][0]) + abs(sheet[i][1] - sheet[self.stat[i]][1])
        # cost = count + self.step
        # return cost
        return count + self.step


class IDAstar:
    # 当白块在9个位置时可以移动的方向，-1代表无法移动
    # w上, d右, s下, a左
    d = [[-1, 1, 3, -1],  # 0
         [-1, 2, 4, 0],  # 1
         [-1, -1, 5, 1],  # 2
         [0, 4, 6, -1],  # 3
         [1, 5, 7, 3],  # 4
         [2, -1, 8, 4],  # 5
         [3, 7, -1, -1],  # 6
         [4, 8, -1, 6],  # 7
         [5, -1, -1, 7]]  # 8
    # 将移动方向的序列转化为'w', 'd', 's', 'a'
    index_to_direct = ['w', 'd', 's', 'a']

    def __init__(self, start, pos, target):
        IDAstar.start = start
        IDAstar.pos = pos
        IDAstar.target = target
        IDAstar.init = Board(start, pos)
        IDAstar.maxdep = 0   # 搜索的最大深度
        IDAstar.path = ""

    def dfs(self, now, lastd):
        # 基于f值的强力剪枝
        if now.cost > self.maxdep:
            return False
        if now.stat == self.target:
            return True
        pos = now.pos
        step = now.step
        for i in range(4):
            # 方向不可走时
            if self.d[pos][i] == -1:
                continue
            stat = copy.deepcopy(now.stat)
            # 0, 1, 2, 3
            # w, d, s, a
            # 上一步为向左，此步则不能向右走老路，其他方向同理。
            if (lastd == -1) or (lastd % 2) != (i % 2) or (lastd == i):
                stat[pos], stat[self.d[pos][i]] = stat[self.d[pos][i]], stat[pos]
                # 构造函数形式：
                # Board(stat, pos, step=0, preboard=None, prepath=[])
                temp = Board(stat, self.d[pos][i], step + 1, now, self.index_to_direct[i])
                # 如果找到最短路径，递归地记录路径
                if self.dfs(temp, i):
                    self.path += temp.prepath
                    return True
        return False

    def IDA(self):
        self.maxdep = self.init.cost
        while not self.dfs(self.init, -1):
            self.maxdep += 1
        self.path = self.path[::-1]
        return self.path


# 将图片填充为正方形
def fill_image(image):
    width, height = image.size
    # 选取长和宽中较大值作为新图片的
    new_image_length = width if width > height else height
    # 生成新图片[白底]
    new_image = Image.new(image.mode, (new_image_length, new_image_length), color='white')
    # 将之前的图粘贴在新图上，居中
    if width > height:  # 原图宽大于高，则填充图片的竖直维度
        new_image.paste(image, (0, int((new_image_length - height) / 2)))  # (x,y)二元组表示粘贴上图相对下图的起始位置
    else:
        new_image.paste(image, (int((new_image_length - width) / 2), 0))
    return new_image


# 切图(n * n)
def cut_image(image, n):
    width, height = image.size
    item_width = int(width / n)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, n):
        for j in range(0, n):
            # print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            # box = np.asarray(box)   # 将切片转换为numpy矩阵
            box_list.append(box)

    image_list = [image.crop(box) for box in box_list]

    return image_list  # 返回numpy矩阵列表


# 保存
def save_images(image_list, content):
    index = 0
    for image in image_list:
        image.save(content + '/' + str(index) + '.jpg', 'JPEG')
        index += 1

def original_partition(dir):
    dirs = './tiles'  # 当前目录下的tiles目录，用于存放所有原图的切片结果
    file_path1 = "original_img"  # 当前目录下的original_img文件夹
    file_path2 = dir + '.jpg'  # 输入original_img文件夹中的jpg文件全名（包括.jpg后缀）
    file_path = os.path.join(file_path1, file_path2)  # 组合成完整的源文件（待切片的图片）路径

    image = Image.open(file_path)  # 打开图片
    # image.show()
    image = fill_image(image)  # 将图片填充为方形
    image_list = cut_image(image, 3)  # 切割图片（3*3）

    # 在tiles文件夹里再建一个文件夹，存放一张原图的所有切片，文件夹的名字与原图文件名（不包括后缀）一样
    dir_path = os.path.join(dirs, dir)  # 组合成完整的目标文件夹路径
    # 判断文件夹是否存在，若不存在则创建目标文件夹
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    save_images(image_list, dir_path)  # 保存切片结果

def img_match(img_base64):
    img = base64.b64decode(img_base64)  # base64编码转字符串
    img = BytesIO(img)  # 字符串转字节流
    pic = Image.open(img)  # 以图片形式打开img
    # 将读取的测试图片保存到本地，同目录下的chaos文件夹中，并命名为integral.jpg
    pic.save('./chaos/integral.jpg', 'JPEG')
    '''
    尝试不将测试图片保存到本地，仅保存在内存中，将字节流改为JPEG格式来打开图像
    失败
    roiImg = Image.open(img)
    imgByteArr = BytesIO()   #创建一个空的Bytes对象
    print(imgByteArr)
    roiImg.save(imgByteArr, 'JPEG') #将roiImg以jpg格式保存在imgByteArr中
    print(imgByteArr)
    print("=====================")
    imgByteArr = imgByteArr.getvalue()
    #print(imgByteArr)
    print("***********************")
    print(roiImg.format, roiImg.mode)
    print("================================")
    pic = Image.open(imgByteArr)
    '''
    # 将原图切分为3*3片，存入img_list列表，并将切片保存到同目录./chaos/discrete文件夹中
    img_list = cut_image(pic, 3)
    save_images(img_list, './chaos/discrete')

    ls_chaos = []  # 存放乱序切片的numpy矩阵的列表
    for root, dirs, files in os.walk("./chaos/discrete"):  # 遍历discrete文件夹
        for file in files:  # 处理该文件夹里的所有文件
            p = Image.open(os.path.join(root, file))  # 合成绝对路径，并打开图像
            p = np.asarray(p)  # 图像转矩阵
            ls_chaos.append(p)  # 将得到的矩阵存入列表
    '''
    #将ls_chaos列表随机打乱
    random.shuffle(ls_chaos)
    '''
    stat = [-1, -1, -1, -1, -1, -1, -1, -1, -1]  # 存放乱序图片的状态，-1代表白块，0~8代表该切片是处于原图中的哪一位置
    dir_path = "./tiles"
    # 遍历同目录中./tiles文件夹中的所有文件夹
    for root, dirs, files in os.walk(dir_path):
        for dir in dirs:
            # k代表状态列表下标，cnt记录当前已匹配上的切片数
            k, cnt = 0, 0
            # tar_stat列表存放目标状态，由于不同原图之间可能存在完全一样的切片，会影响tar_stat的最终结果
            # 因此每次与新的一张原图比较前，将tar_stat初始化为全-1
            tar_stat = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
            # 从ls_chaos列表（即乱序切片的numpy矩阵列表）中，逐个与原图库中的切片比较
            for i in ls_chaos:
                # index用于指示乱序的切片在原图的哪一位置
                index = 0
                # 遍历存放原图切片的文件夹中的所有文件（即，原图切片）
                for root, dirs, files in os.walk(os.path.join(dir_path, dir)):
                    for j in files:
                        # 用os.path.join()拼接出文件的绝对路径，然后打开该文件（图片）
                        j = Image.open(os.path.join(root, j))
                        j = np.asarray(j)  # 将原图切片转换为numpy矩阵
                        if (i == j).all():  # 判断两个矩阵是否完全相同
                            stat[k] = index
                            tar_stat[index] = index
                            cnt += 1
                            break
                        index += 1
                    k += 1
            # 若已有8个切片匹配上则说明匹配到了原图
            if cnt > 7:
                print("原图是:", dir)  # 打印原图名称
                break
    if cnt <8:
        print("没找到原图QAQ")
    # 遍历初始状态列表，获得白块的初始位置
    for i in range(len(stat)):
        if stat[i] < 0:
            blank = i
            break
    # 返回初始状态（列表）、空白块位置、目标状态（列表）
    return stat, blank, tar_stat



if __name__ == '__main__':
    uu=input("请输入URL:")
    url = uu
    data={'teamid':43,'token':'db89ae69-2536-4f96-91b0-d1c998a7850b'}
    data = json.dumps(data)  # 有的时候data需要时json类型的
    headers = {'content-type': "application/json"}
    #headers = {'content-type': application / json}  # 一种请求头，需要携带
    resp = rq.post(url=url,data=data,headers=headers)
    a=json.loads(resp.text)
    b=a.get("data")
    print("第几步进行强制转换: ", b.get("step"))
    print("调换的图片编号: ", b.get("swap"))
    print("题目标识: ", a.get("uuid"))
    print("是否成功：",a.get("success"))
    start, pos, end = img_match(b.get("img"))
    print("初始状态: ", start)  # 输出乱序图片的状态，也即拼图游戏的初始状态（-1代表白块，0~8代表该切片是处于原图中的哪一位置）
    print("白块位置: ", pos)    # 输出空白块在乱序图中的位置
    print("目标状态: ", end)  # 输出目标状态
    # print(type(start), type(end))
    solve = IDAstar(start, pos, end)
    path = solve.IDA()
    print(path, "路径长度: ", len(path))
