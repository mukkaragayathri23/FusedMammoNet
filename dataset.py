import pandas as pd
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch
import numpy as np
import os
import tensorflow as tf

class CustomDataset(Dataset):
    def __init__(self, filenames, transfer_learning=False):
        self.filenames = filenames
        self.transfer_learning = transfer_learning

        self.images, self.labels = self.read_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = np.stack([image, image, image], axis=0)

        # Convert to PyTorch tensor
        image = torch.from_numpy(image).float()
        label = torch.tensor(label, dtype=torch.long) # Add batch dimension
        return image, label

    def read_data(self):
        images, labels = [], []

        for file in self.filenames:
            # Use TensorFlow to load TFRecord data
            raw_dataset = tf.data.TFRecordDataset(file)

            for raw_record in raw_dataset:
                example = tf.io.parse_single_example(raw_record, {
                    'label': tf.io.FixedLenFeature([], tf.int64),
                    'label_normal': tf.io.FixedLenFeature([], tf.int64),
                    'image': tf.io.FixedLenFeature([], tf.string),
                })

                # Decode the image feature
                image = tf.io.decode_raw(example['image'], tf.uint8).numpy()
                image = image.reshape([299,299])  # Adjust dimensions accordingly
                image = cv2.resize(image, (224,224))  # Adjust dimensions accordingly

                images.append(image / 255)
                labels.append(example['label'].numpy())

        return images, labels                         
def create_loaders(filepath):
    root_dir = os.path.abspath(filepath)
    filenames = [
        os.path.join(root_dir, 'training10_0', 'training10_0.tfrecords')
    ]

    # Create dataset and dataloaders
    custom_dataset = CustomDataset(filenames, transfer_learning=False)

    # Split dataset
    train_size = int(0.6 * len(custom_dataset))
    val_size = int(0.2 * len(custom_dataset))
    test_size = len(custom_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        custom_dataset, [train_size, val_size, test_size]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_dataloader,val_dataloader,test_dataloader