import numpy as np
import torch.utils.data as data
import cv2
import pandas as pd
import os, torch
import image_utils
import argparse, random

class Dataset(data.Dataset):
    def __init__(self, raf_path, train, transform=None, basic_aug=False):
        self.train = train
        self.transform = transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        if self.train:
            df_train = pd.read_csv(
                os.path.join(self.raf_path, 'EmoLabel'),
                sep=' ',
                header=None)

            self.file_paths = []
            label = []

            for index, row in df_train.iterrows():
                file_name = row.iloc[NAME_COLUMN]
                labels = row.iloc[LABEL_COLUMN] - 1
                file_name = file_name.split(".")[0]

                file_path = os.path.join(self.raf_path, 'Image', file_name)
                self.file_paths.append(file_path)
                label.append(labels)

            self.label = np.array(label)

        else:
            df_val = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel'),
                                 sep=' ',
                                 header=None)

            self.file_paths = []
            label = []

            for index, row in df_val.iterrows():
                file_name = row.iloc[NAME_COLUMN]
                labels = row.iloc[LABEL_COLUMN] - 1
                file_name = file_name.split(".")[0]

                file_path = os.path.join(self.raf_path, 'Image', file_name)
                self.file_paths.append(file_path)
                label.append(labels)

            self.label = np.array(label)
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        # augmentation
        if self.train:
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label