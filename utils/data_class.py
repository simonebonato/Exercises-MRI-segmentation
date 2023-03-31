import os
import torchio as tio
from torch.utils.data import DataLoader, random_split
import gdown
from pathlib import Path
import numpy as np


class MedicalDecathlonDataModule():
    def __init__(self, task, google_id, batch_size, train_val_ratio):
        self.task = task
        self.google_id = google_id
        self.batch_size = batch_size
        self.dataset_dir = Path(task)
        self.train_val_ratio = train_val_ratio
        self.subjects = None
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def get_max_shape(self, subjects):
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)

    def download_data(self):
        if not self.dataset_dir.is_dir():
            print("Downloading dataset...\n")
            url = f"https://drive.google.com/uc?id={self.google_id}"
            output = f"{self.task}.tar"
            gdown.download(url, output, quiet=False)
            os.system(f"tar xf {output}")

        def get_niis(d):
            return sorted(p for p in d.glob("*.nii*") if not p.name.startswith("."))

        image_training_paths = get_niis(self.dataset_dir / "imagesTr")
        label_training_paths = get_niis(self.dataset_dir / "labelsTr")
        image_test_paths = get_niis(self.dataset_dir / "imagesTs")
        return image_training_paths, label_training_paths, image_test_paths

    def prepare_data(self):
        (
            image_training_paths,
            label_training_paths,
            image_test_paths,
        ) = self.download_data()

        self.subjects = []
        for image_path, label_path in zip(image_training_paths, label_training_paths):
            # 'image' and 'label' are arbitrary names for the images
            subject = tio.Subject(
                image=tio.ScalarImage(image_path), label=tio.LabelMap(label_path)
            )
            self.subjects.append(subject)

        self.test_subjects = []
        for image_path in image_test_paths:
            subject = tio.Subject(image=tio.ScalarImage(image_path))
            self.test_subjects.append(subject)

    def get_preprocessing_transform(self):
        preprocess = tio.Compose(
            [
                tio.RescaleIntensity((-1, 1)),
                tio.CropOrPad(self.get_max_shape(self.subjects + self.test_subjects)),
                tio.EnsureShapeMultiple(8),  # for the U-Net
                tio.OneHot(),
            ]
        )
        return preprocess

    def get_augmentation_transform(self):
        augment = tio.Compose(
            [
                tio.RandomAffine(),
                tio.RandomGamma(p=0.5),
                tio.RandomNoise(p=0.5),
                tio.RandomMotion(p=0.1),
                tio.RandomBiasField(p=0.25),
            ]
        )
        return augment

    def setup(self, stage=None):
        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(self.subjects, splits)

        self.preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        self.transform = tio.Compose([self.preprocess, augment])

        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)
        self.test_set = tio.SubjectsDataset(
            self.test_subjects, transform=self.preprocess
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size)
