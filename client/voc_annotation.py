import os
import random
import xml.etree.ElementTree as ET
import numpy as np
from utils.utils import get_classes

# --------------------------------------------------------------------------------------------------------------------------------#
#   annotation_mode用于指定该文件运行时计算的内容
#   annotation_mode为0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面索引的txt以及训练用的2007_train.txt、2007_val.txt
#   annotation_mode为1代表获得VOCdevkit/VOC2007/ImageSets里面的txt
#   annotation_mode为2代表获得训练用的2007_train.txt、2007_val.txt
# --------------------------------------------------------------------------------------------------------------------------------#
annotation_mode     = 2
# -------------------------------------------------------------------#
#   必须要修改，用于生成2007_train.txt、2007_val.txt的目标信息
#   与训练和预测所用的classes_path一致即可
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
#   仅在annotation_mode为0和2的时候有效
# -------------------------------------------------------------------#
classes_path        = '../model_data/voc_classes.txt'
# --------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1
#   仅在annotation_mode为0和1的时候有效
# --------------------------------------------------------------------------------------------------------------------------------#
trainval_percent    = 0.9
train_percent       = 0.9
# -------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
# -------------------------------------------------------#
VOCdevkit_path = '../VOCdevkit'

VOCdevkit_set  = [('2007', 'train'), ('2007', 'val')]
classes, _     = get_classes(classes_path)

# -------------------------------------------------------#
#   统计目标数量
# -------------------------------------------------------#
photo_nums = np.zeros(len(VOCdevkit_set))
nums = np.zeros(len(classes))

# ------------------------------------------------------#
# 定义方法处理PASCAL VOC格式的XML标注文件，
# 用于从文件中提取目标对象的边界框和类别信息，并可能过滤掉一些困难的对象或不属于指定类别的对象
# ------------------------------------------------------#


def convert_annotation(year, image_id, list_file):
    # 构建文件路径，指向VOCdevkit_path/VOC2007/Annotation/xxx.xml文件，其中year和image_id是方法输入的参数
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml' % (year, image_id)), encoding='utf-8')
    # 打开文件，返回一个XML对象
    tree = ET.parse(in_file)
    # 获取XML对象的根元素，通常为‘annotation’
    root = tree.getroot()

    # 遍历根元素root下的object元素，每个object元素代表一个识别目标对象
    for obj in root.iter('object'):
        # 先判断该目标识别的难易程度，0为容易识别，1是难以识别
        difficult = 0
        # 如果object元素中difficult元素非空,则将该元素值赋予difficult
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        # 将object下的类别元素值赋予cls
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        # 找到该类别在类别列表中的索引值
        cls_id = classes.index(cls)
        # 在object元素下找到bndbox元素，并返回该元素
        xmlbox = obj.find('bndbox')
        # 提取bndbox元素下的左上角和右下角坐标的值，先转化为浮点数，在转化为整数
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        # 每个object元素之间用‘空格’隔开，每个object内的坐标值和类别值用‘逗号’隔开
        list_file.write(" " + ",".join([str(a) for a in b]) + "," + str(cls_id))
        # 相应类别的计数值加1
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1


if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")
    # 先进行索引文件txt的生成，对于模式0和1均需要生成该文件
    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        # 构建xml文件的存放目录
        xmlfilepath = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        # 构建索引文件txt的存放目录
        saveBasepath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')
        # 将xml文件夹下的文件和子目录的名称以字符串形式存储在列表中，注意这里的名称包含文件类型后缀，即.xml
        temp_xml = os.listdir(xmlfilepath)
        total_xml = []
        # 将后缀包含.xml的筛选出来
        for xml in temp_xml:
            if xml.endswith('.xml'):
                total_xml.append(xml)
        # 标注文件总数
        num = len(total_xml)
        # 形成总体索引列表
        olist = range(num)
        # 计算训练验证集的样本数量，计算训练集的样本数量
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        # 在总体索引列表中采样，选取tv个数据，形成训练验证集索引列表
        trainval_list = random.sample(olist, tv)
        # 在训练验证集索引列表中采样，选取tr个数据，形成训练集索引列表
        train_list = random.sample(trainval_list, tr)

        print("train and val size", tv)
        print("train size", tr)
        # 打开文件对象
        ftrainval = open(os.path.join(saveBasepath, 'trainval.txt'), 'w')
        ftrain    = open(os.path.join(saveBasepath, 'train.txt'), 'w')
        fval      = open(os.path.join(saveBasepath, 'val.txt'), 'w')
        ftest     = open(os.path.join(saveBasepath, 'test.txt'), 'w')

        # 遍历总体索引列表，将元素分发给各个集合中
        for i in olist:
            # olist每个元素均为字符串，剔除后四位的后缀
            name = total_xml[i][:-4] + '\n'
            if i in trainval_list:
                ftrainval.write(name)
                if i in train_list:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate txt in ImageSets done.")

    # 在生成训练所用的总体索引文件，文件内每一行包括图片位置，真实框参数和类别
    # 对于模式0和模式2均需要进行
    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        type_index = 0
        for year, image_set in VOCdevkit_set:
            # 打开相应image_set的索引文件，读取内容，删除字符串首位空格，并按照换行符分隔，返回一个字符串列表
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)), encoding='utf-8').read().strip().split()
            # 按照image_set内容建立最终要输出的文件
            list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                # 对于每个索引id，将对应图片的位置添加到索引文件中,注意这里添加图片的绝对路径
                list_file.write('%s/VOC%s/JPEGImages/%s.jpg' % (os.path.abspath(VOCdevkit_path), year, image_id))
                # 将该图片的标注信息（真实框位置和类别）添加到索引文件中
                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for train done.")


        def printTable(List1, List2):
            for i in range(len(List1[0])):
                print("|", end=' ')
                for j in range(len(List1)):
                    print(List1[j][i].rjust(int(List2[j])), end=' ')
                    print("|", end=' ')
                print()


        str_nums = [str(int(x)) for x in nums]
        tableData = [
            classes, str_nums
        ]
        colWidths = [0] * len(tableData)
        len1 = 0
        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print(
                "训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。")

        if np.sum(nums) == 0:
            print(
                "在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print(
                "在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print(
                "在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("（重要的事情说三遍）。")
