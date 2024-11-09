import os
import shutil  # 复制文件
import random
import numpy as np


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


suffix_xml = '.xml'
suffix_jpeg = '.jpg'

car_train_path = 'D:\\Dataset\\PASCAL_VOC_2007\\VOCtrainval_06-Nov-2007_2\\VOCdevkit\\VOC2007\\ImageSets\\Main\\car_train.txt'
car_val_path = 'D:\\Dataset\\PASCAL_VOC_2007\\VOCtrainval_06-Nov-2007_2\\VOCdevkit\\VOC2007\\ImageSets\\Main\\car_val.txt'

# 包含原始 VOC 注释和 JPEG 图像文件的目录。
voc_annotation_dir = 'D:\\Dataset\\PASCAL_VOC_2007\\VOCtrainval_06-Nov-2007_2\\VOCdevkit\\VOC2007\\Annotations\\'
voc_jpeg_dir = 'D:\\Dataset\\PASCAL_VOC_2007\\VOCtrainval_06-Nov-2007_2\\VOCdevkit\\VOC2007\\JPEGImages'

# 保存所选汽车图像和注释的根目录。
car_root_dir = 'D:\\Dataset\\PASCAL_VOC_2007\\voc_car\\'


def parse_train_val(data_path):
    """提取指定类别的图像"""
    samples = []
    with open(data_path, 'r') as file:
        lines = file.readlines()  # 读取文件的每一行
        for line in lines:
            res = line.strip().split(" ")  # 划分行
            # print(res)
            # 如果标志为1，表示汽车
            if len(res) == 3 and int(res[2]) == 1:
                samples.append(res[0])
    return np.array(samples)


def sample_train_val(samples):
    """随机采样samples中图像的 1/10"""
    for name in ["train", "val"]:
        dataset = samples[name]
        length = len(dataset)

        random_samples = random.sample(range(length), int(length / 10))

        new_dataset = dataset[random_samples]
        samples[name] = new_dataset

    return samples


def save_car(car_samples, data_root_dir, data_annotation_dir, data_jpeg_dir):
    """保存到所需目录"""
    # 遍历每个图像名称
    for sample_name in car_samples:
        # 对每个car_samples中的每个sample_name，构建xml注释和jpeg图像的源路径和目标路径
        src_annotation_path = os.path.join(voc_annotation_dir, sample_name + suffix_xml)
        dst_annotation_path = os.path.join(data_annotation_dir, sample_name + suffix_xml)
        shutil.copyfile(src_annotation_path, dst_annotation_path)

        src_jpeg_path = os.path.join(voc_jpeg_dir, sample_name + suffix_jpeg)
        dst_jpeg_path = os.path.join(data_jpeg_dir, sample_name + suffix_jpeg)
        shutil.copyfile(src_jpeg_path, dst_jpeg_path)

    csv_path = os.path.join(data_root_dir, 'car.csv')
    np.savetxt(csv_path, np.array(car_samples), fmt='%s')


if __name__ == '__main__':
    samples = {
        "train": parse_train_val(car_train_path),
        "val": parse_train_val(car_val_path),
    }
    # print(samples)

    check_dir(car_root_dir)
    for name in ["train", "val"]:
        data_root_dir = os.path.join(car_root_dir, name)
        data_annotation_dir = os.path.join(data_root_dir, "Annotations")
        data_jpeg_dir = os.path.join(data_root_dir, "JPEGImages")

        # print(data_root_dir)
        # print(data_annotation_dir)
        # print(data_jpeg_dir)

        check_dir(data_root_dir)
        check_dir(data_annotation_dir)
        check_dir(data_jpeg_dir)

        save_car(samples[name], data_root_dir, data_annotation_dir, data_jpeg_dir)

    print("done")
