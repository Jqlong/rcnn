import numpy as np
import os
import cv2
from torch.utils.data import Dataset

from utils.create_finetune_data import parse_car_csv


class CustomClassifierDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        samples = parse_car_csv(root_dir)
        jpeg_images = []  # 存储加载的图像
        annotation_list = []  # 存储边界框和关联的标签

        for idx, sample_name in enumerate(samples):
            jpeg_image = cv2.imread(
                os.path.join(root_dir, "JPEGImages", sample_name + ".jpg")  # 找到对应的图片
            )

            positive_annotation_path = os.path.join(
                root_dir, "Annotations", sample_name + "_1.csv"  # 找到对应的正文件
            )
            positive_annotations = np.loadtxt(positive_annotation_path, dtype=np.int64, delimiter=" ")
            positive_annotations = (
                [positive_annotations]
                if positive_annotations.ndim == 1
                else positive_annotations  # 边界
            )
            for annotation in positive_annotations:
                annotation_list.append(
                    {
                        "rect": annotation,
                        "image_id": idx,
                        "target": 1,
                    }
                )

            negative_annotation_path = os.path.join(
                root_dir, "Annotations", sample_name + "_0.csv"
            )
            negative_annotations = np.loadtxt(negative_annotation_path, dtype=np.int64, delimiter=" ")
            negative_annotations = (
                [negative_annotations]
                if negative_annotations.ndim == 1
                else negative_annotations
            )
            for annotation in negative_annotations:
                annotation_list.append(
                    {
                        "rect": annotation,
                        "image_id": idx,
                        "target": 0,
                    }
                )

            jpeg_images.append(jpeg_image)

        self.transform = transform
        self.jpeg_images = jpeg_images
        self.annotation_list = annotation_list

    def get_positives(self):
        # Filter and return all annotations where target is 1
        positive_list = [ann for ann in self.annotation_list if ann["target"] == 1]
        return positive_list

    def get_negatives(self):
        # Filter and return all annotations where target is 0
        negative_list = [ann for ann in self.annotation_list if ann["target"] == 0]
        return negative_list

    def get_positive_num(self):
        positive_list = [ann for ann in self.annotation_list if ann["target"] == 1]
        return len(positive_list)

    def get_negative_num(self):
        negative_list = [ann for ann in self.annotation_list if ann["target"] == 0]
        return len(negative_list)

    def set_negative_list(self, new_negative_list):
        """
        This method replaces the current negative list with a new one.
        """
        # self.negative_list = new_negative_list

        # Update annotation_list to reflect new negative samples
        self.annotation_list = [
            ann for ann in self.annotation_list if ann["target"] == 1
        ]  # keep positive samples
        self.annotation_list.extend(new_negative_list)

    def __getitem__(self, index: int):
        annotation = self.annotation_list[index]
        xmin, ymin, xmax, ymax = annotation["rect"]
        image_id = annotation["image_id"]
        target = annotation["target"]
        image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]

        if self.transform:
            image = self.transform(image)

        return image, target  # 得到候选区的图片

    def __len__(self):
        return len(self.annotation_list)