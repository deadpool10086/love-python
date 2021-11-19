""" makeVerification.py 文件"""

import random
import os
from PIL import Image,ImageDraw,ImageFont,ImageFilter

# 设置图片宽高
width = 60
height = 20
"""
"""

def getRandomColor(is_light = True):
    """
    生成随机颜色
    :param is_light: 为了设置浅色和深色
    :return:  (r, g, b)
    """
    r = random.randint(0, 127) +int(is_light)* 128
    g = random.randint(0, 127) +int(is_light)* 128
    b = random.randint(0, 127) +int(is_light)* 128
    return (r, g, b)

# 这里全部数据
def getRandomChar():
    random_num = str(random.randint(0, 9))      # 数字 0~9
    random_lower = chr(random.randint(97, 122)) # 小写字母a~z
    random_upper = chr(random.randint(65, 90))  # 大写字母A~Z
    random_char = random.choice([random_num, random_upper, random_lower])
    return random_char

def drawLine(draw):
    """
    随机生成4个干扰线，然后每个设置随机颜色
    """
    for i in range(4):
        x1 = random.randint(0, width)
        x2 = random.randint(0, width)
        y1 = random.randint(0, height)
        y2 = random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill=getRandomColor(is_light=True))

def drawPoint(draw):
    """
    随机生成80个干扰点，然后每个设置随机颜色
    """
    for i in range(80):
        x = random.randint(0, width)
        y = random.randint(0, height)
        draw.point((x,y), fill=getRandomColor(is_light=True))

def createImg(folder):
    # 随机生成一个颜色为背景色颜色
    bg_color = getRandomColor(is_light=True)

    # 创建一张随机背景色的图片
    img = Image.new(mode="RGB", size=(width, height), color=(255,255,255))

    # 获取图片画笔，用于描绘字
    draw = ImageDraw.Draw(img)

    # 修改字体
    font = ImageFont.truetype(font="arial.ttf", size=18)

    # 保存图片的名字
    file_name = ''

    # 这里生成4位数字，就循环4次
    for i in range(4):
        # 随机生成4种字符+4种颜色
        random_txt = getRandomChar()
        txt_color = getRandomColor(is_light=False)

        # 避免文字颜色和背景色一致重合
        while txt_color == bg_color:
            txt_color = getRandomColor(is_light=False)
        # 根据坐标填充文字
        draw.text((2 + 15 * i, 0), text=random_txt, fill=txt_color, font=font)
        file_name +=random_txt
    # 画干扰线和点
    # drawLine(draw)
    drawPoint(draw)
    print(file_name)

    # 打开图片操作，并保存在当前文件夹下
    with open("./{}/{}.png".format(folder,file_name), "wb") as f:

        img.save(f, format="png")

if __name__ == '__main__':
    # 创建num张验证码
    num = 1000

    # 创建train和test文件夹
    os.path.exists('train') or  os.makedirs('train')
    os.path.exists('test') or os.makedirs('test')

    # 每个文件夹创建num个
    for i in range(num):
        createImg('train')
        createImg('test')


