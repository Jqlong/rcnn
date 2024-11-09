import os.path
import shutil
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import xmltodict
import selectivesearch
from utils.pascal_voc_car import check_dir

"""创建微调数据"""


def parse_xml(xml_path):
    """
    アノテーションのバウンディングボックスの座標を返すためにxmlファイルをパースする
    """
    with open(xml_path, "rb") as f:
        xml_dict = xmltodict.parse(f)

        bndboxs = list()
        objects = xml_dict["annotation"]["object"]
        if isinstance(objects, list):
            for obj in objects:
                obj_name = obj["name"]
                difficult = int(obj["difficult"])
                if "car".__eq__(obj_name) and difficult != 1:
                    bndbox = obj["bndbox"]
                    bndboxs.append(
                        (
                            int(bndbox["xmin"]),
                            int(bndbox["ymin"]),
                            int(bndbox["xmax"]),
                            int(bndbox["ymax"]),
                        )
                    )
        elif isinstance(objects, dict):
            obj_name = objects["name"]
            difficult = int(objects["difficult"])
            if "car".__eq__(obj_name) and difficult != 1:
                bndbox = objects["bndbox"]
                bndboxs.append(
                    (
                        int(bndbox["xmin"]),
                        int(bndbox["ymin"]),
                        int(bndbox["xmax"]),
                        int(bndbox["ymax"]),
                    )
                )
        else:
            pass

        return np.array(bndboxs)


def iou(pred_box, target_box):
    """
    候補となる提案とラベル付きバウンディングボックスのIoUを計算する
    :param pred_box: size [4].
    :param target_box: size [N, 4] :return: [N].
    :return: [N］
    """
    if len(target_box.shape) == 1:
        target_box = target_box[np.newaxis, :]

    xA = np.maximum(pred_box[0], target_box[:, 0])
    yA = np.maximum(pred_box[1], target_box[:, 1])
    xB = np.minimum(pred_box[2], target_box[:, 2])
    yB = np.minimum(pred_box[3], target_box[:, 3])

    intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)

    boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBArea = (target_box[:, 2] - target_box[:, 0]) * (
        target_box[:, 3] - target_box[:, 1]
    )

    scores = intersection / (boxAArea + boxBArea - intersection)
    return scores

def compute_ious(rects, bndboxs):
    iou_list = list()
    for rect in rects:
        scores = iou(rect, bndboxs)
        iou_list.append(max(scores))
    return iou_list


def parsec_annotation_jpeg(annotation_path, jpeg_path, gs):
    # 加载图像并执行选择性搜索
    img = cv2.imread(jpeg_path)

    selectivesearch.config(gs, img, strategy='q')
    rects = selectivesearch.get_rects(gs)  # 生成区域建议

    # 取得边框
    bndboxes = parse_xml(annotation_path)  # 读取文件，获取边界框

    # 获取最大的边界框
    maximum_bndbox_size = 0
    for bndbox in bndboxes:
        xmin, ymin, xmax, ymax = bndbox
        bndbox_size = (ymax - ymin) * (xmax - xmin)
        if bndbox_size > maximum_bndbox_size:
            maximum_bndbox_size = bndbox_size

    iou_list = compute_ious(rects, bndboxes)  # 对每个候选框计算IOU分数，并选择最大的值

    # 初始化正负区域
    positive_list = list()
    negative_list = list()
    for i in range(len(iou_list)):
        xmin, ymin, xmax, ymax = rects[i]
        rect_size = (ymax - ymin) * (xmax - xmin)

        iou_score = iou_list[i]
        if iou_list[i] >= 0.5:
            # 正样本
            positive_list.append(rects[i])
        if 0 < iou_list[i] < 0.5 and rect_size > maximum_bndbox_size / 5.0:
            # 负样本
            negative_list.append(rects[i])
        else:
            pass

    return positive_list, negative_list

def parse_car_csv(csv_dir):
    csv_path = os.path.join(csv_dir, 'car.csv')
    samples = np.loadtxt(csv_path, dtype="unicode")
    return samples


if __name__ == "__main__":
    car_root_dir = "D:\\Dataset\\PASCAL_VOC_2007\\voc_car\\"
    finetune_root_dir = "D:\\Dataset\\PASCAL_VOC_2007\\finetune_car\\"
    check_dir(finetune_root_dir)

    gs = selectivesearch.get_selective_search()
    for name in ["train", "val"]:
        src_root_dir = os.path.join(car_root_dir, name)
        src_annotation_dir = os.path.join(src_root_dir, "Annotations")
        src_jpeg_dir = os.path.join(src_root_dir, "JPEGImages")

        dst_root_dir = os.path.join(finetune_root_dir, name)
        dst_annotation_dir = os.path.join(dst_root_dir, "Annotations")
        dst_jpeg_dir = os.path.join(dst_root_dir, "JPEGImages")
        check_dir(dst_root_dir)
        check_dir(dst_annotation_dir)
        check_dir(dst_jpeg_dir)

        total_num_positive = 0
        total_num_negative = 0

        samples = parse_car_csv(src_root_dir)

        src_csv_path = os.path.join(src_root_dir, "car.csv")
        dst_csv_path = os.path.join(dst_root_dir, "car.csv")
        shutil.copyfile(src_csv_path, dst_csv_path)
        for sample_name in samples:
            since = time.time()

            src_annotation_path = os.path.join(src_annotation_dir, sample_name + ".xml")
            src_jpeg_path = os.path.join(src_jpeg_dir, sample_name + ".jpg")

            positive_list, negative_list = parsec_annotation_jpeg(
                src_annotation_path, src_jpeg_path, gs
            )
            total_num_positive += len(positive_list)
            total_num_negative += len(negative_list)

            dst_annotation_positive_path = os.path.join(
                dst_annotation_dir, sample_name + "_1" + ".csv"
            )
            dst_annotation_negative_path = os.path.join(
                dst_annotation_dir, sample_name + "_0" + ".csv"
            )
            dst_jpeg_path = os.path.join(dst_jpeg_dir, sample_name + ".jpg")

            shutil.copyfile(src_jpeg_path, dst_jpeg_path)

            np.savetxt(
                dst_annotation_positive_path,
                np.array(positive_list),
                fmt="%d",
                delimiter=" ",
            )
            np.savetxt(
                dst_annotation_negative_path,
                np.array(negative_list),
                fmt="%d",
                delimiter=" ",
            )

            time_elapsed = time.time() - since
            print(
                "parse {}.png in {:.0f}m {:.0f}s".format(
                    sample_name, time_elapsed // 60, time_elapsed % 60
                )
            )
        print("%s positive num: %d" % (name, total_num_positive))
        print("%s negative num: %d" % (name, total_num_negative))
    print("done")


