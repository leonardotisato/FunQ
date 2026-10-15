from torchvision.io import read_image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from typing import Tuple, List, Dict, Any, Callable, Optional
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from PIL import Image
import albumentations as A
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import math
import cv2
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from omegaconf import DictConfig
import time

def safe_pil_read(path, retries=10, delay=1.0):
    # WSL2 can drop file handles under heavy IO load, causing "Premature end of JPEG file"
    for i in range(retries):
        try:
            img = Image.open(path).convert("RGB")
            img.load()  # Force load to catch truncated files
            return img
        except Exception:
            time.sleep(delay)
    # Final attempt that will raise the actual error if it fails
    img = Image.open(path).convert("RGB")
    img.load()
    return img

def safe_cv2_read(path, flags=None, retries=10, delay=1.0):
    # Retry loop for cv2.imread which returns None on silent WSL2 IO drops
    for i in range(retries):
        img = cv2.imread(path, flags) if flags is not None else cv2.imread(path)
        if img is not None:
            return img
        time.sleep(delay)
    return cv2.imread(path, flags) if flags is not None else cv2.imread(path)


def trim_fundus_black_border(
    image: np.ndarray,
    threshold: int = 8,
    pad_ratio: float = 0.01,
    min_pad_px: int = 4,
) -> np.ndarray:
    """Trim only the clearly non-informative outer border of a fundus image.

    The crop is *not* lesion-aware and uses only image content. During local
    inspection against the lesion masks, a plain non-black crop could exclude a
    handful of very dark edge pixels, so we keep a small safety pad around the
    detected content box. This makes the rule conservative while still
    improving information density before resizing to a lower resolution.
    """

    if image.ndim != 3:
        return image

    gray = image.mean(axis=2)
    ys, xs = np.where(gray > threshold)
    if len(xs) == 0 or len(ys) == 0:
        return image

    h, w = image.shape[:2]
    pad = max(min_pad_px, int(round(min(h, w) * pad_ratio)))
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, h)

    return image[y0:y1, x0:x1]

class FundusClsDataset(Dataset):
    def __init__(
        self,
        data_csv,
        train=True,
        transform=None,
        preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        if isinstance(data_csv, str):
            self.data_csv = pd.read_csv(data_csv)
        else:
            self.data_csv = data_csv
        self.train = train  # Flag indicating if the dataset is used for training
        self.transform = transform  # Optional transforms
        self.preprocess = preprocess
    
    def __len__(self):
        """Returns the length of the dataset (number of rows in the CSV)."""
        return len(self.data_csv.index)
    
    def __getitem__(self, idx):
        """Fetches the image, mask, and applies transformations. Constructs the multi-channel mask."""
        
        if idx not in self.data_csv.index:
            raise IndexError(f"Index {idx} not found in the dataset.")
        
        # Load image and mask using the paths from the CSV
        label = self.data_csv.iloc[idx]['label']  # Fetch the label (0: Healthy (considered background), 1: nevi, 2: UM, 3: CHRPE)
        img_path = str(self.data_csv.iloc[idx]['image']).strip()
        img = safe_pil_read(img_path)
        img = np.array(img) 

        if self.preprocess is not None:
            img = self.preprocess(img)
        
        # -1 otherwise cross entropy loss does not works

        if self.transform:
            augmented = self.transform(image=img)  # Pass as named argument
            x_img = augmented['image']  # Extract transformed image

        label = torch.tensor(label, dtype=torch.long)
        img_tensor = transforms.ToTensor()(np.float32(x_img))
        #print(f"values img: {img_tensor.min(), img_tensor.max()}")

        return img_tensor, label

# --- DATASET ZOOM LESION ---


class BaseLesionZoomDataset(Dataset):
    def __init__(self, csv_file, transform=None, dilation_percentage=0.5, is_training=True, random_seed=42):
        self.transform = transform
        self.dilation_percentage = dilation_percentage
        self.is_training = is_training
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.data_csv = csv_file if isinstance(csv_file, pd.DataFrame) else pd.read_csv(csv_file)
        required_columns = ['image', 'mask', 'label']
        if not all(col in self.data_csv.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")

    def __len__(self):
        return len(self.data_csv)

    def _dilate_mask(self, mask):
        if self.dilation_percentage == 0.0:
            return mask
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            return mask
        linear_size = math.sqrt(np.count_nonzero(mask))
        dilation_pixels = int(self.dilation_percentage * linear_size * 0.5)
        dilation_pixels = max(3, dilation_pixels)
        kernel = np.ones((dilation_pixels, dilation_pixels), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)

    def _get_lesion_bbox(self, mask):
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            return None
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        return (x_min, y_min, x_max, y_max)

    def _zoom_healthy_image(self, image, mask):
        h, w = image.shape[:2]
        zoom_factor = 0.05 + self.dilation_percentage * 0.03
        crop_w, crop_h = int(zoom_factor * w), int(zoom_factor * h)
        min_dim = min(crop_w, crop_h)
        padding = int(min_dim * 0.5)
        crop_w = min(w, crop_w + 2 * padding)
        crop_h = min(h, crop_h + 2 * padding)
        center_x, center_y = w // 2, h // 2
        x1 = max(center_x - crop_w // 2, 0)
        x2 = min(center_x + crop_w // 2, w)
        y1 = max(center_y - crop_h // 2, 0)
        y2 = min(center_y + crop_h // 2, h)
        return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    def _crop_around_lesion(self, image, mask, mask_original):
        bbox = self._get_lesion_bbox(mask)
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            img_c, mask_c = self._zoom_healthy_image(image, mask)
            _, mask_orig_c = self._zoom_healthy_image(image, mask_original)
            return img_c, mask_c, mask_orig_c
        else:
            x_min, y_min, x_max, y_max = bbox
            return (image[y_min:y_max, x_min:x_max],
                    mask[y_min:y_max, x_min:x_max],
                    mask_original[y_min:y_max, x_min:x_max])

    def _load_image_and_mask(self, idx):
        row = self.data_csv.iloc[idx]
        img_path = str(row['image']).strip()
        mask_path = str(row['mask']).strip()
        label = row['label']
        image = safe_cv2_read(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if label == 0:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            mask_original = mask.copy()
        else:
            binary_mask = safe_cv2_read(mask_path, cv2.IMREAD_GRAYSCALE)
            if binary_mask is None:
                raise ValueError(f"Failed to load mask: {mask_path}")
            mask_original = (binary_mask > 0).astype(np.uint8) * label
            if self.dilation_percentage > 0:
                binary_mask = self._dilate_mask(binary_mask)
            mask = (binary_mask > 0).astype(np.uint8) * label
        return image, mask, mask_original, label

class FundusClsDatasetZoom(BaseLesionZoomDataset):
    def __getitem__(self, idx):
        image, mask, mask_original, label = self._load_image_and_mask(idx)
        image, mask, mask_original = self._crop_around_lesion(image, mask, mask_original)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        image = np.array(image) 
        mask_original = cv2.resize(mask_original, (224, 224), interpolation=cv2.INTER_NEAREST)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        img_tensor = transforms.ToTensor()(np.float32(image))
        mask_tensor = torch.from_numpy(mask_original).long()
        #print(f"values img: {img_tensor.min(), img_tensor.max()}")
        return img_tensor, torch.tensor(label, dtype=torch.long)

class PairedFundusDataset(Dataset):
    """
    Returns (img_teacher, img_student, label, image_name)
    Ensures that teacher and student datasets are aligned.
    """
    def __init__(
        self,
        df,
        teacher_transform,
        student_transform,
        dilation_percentage=0.5,
        student_train=True,
        teacher_train=True
    ):

        self.ds_teacher = FundusClsDatasetZoom(
            df,
            transform=teacher_transform,
            dilation_percentage=dilation_percentage  
        )
        self.ds_student = FundusClsDataset(
            df,
            train=student_train,
            transform=student_transform
        )

        assert len(self.ds_teacher) == len(self.ds_student), "Teacher/Student datasets length mismatch."
 
        self.data_csv = df.reset_index(drop=True)

    def __len__(self):
        return len(self.ds_student)

    def __getitem__(self, idx):
        img_t, label_t = self.ds_teacher[idx]
        img_s, label_s = self.ds_student[idx]
   
        if label_t != label_s:
            raise RuntimeError(f"Label mismatch at idx {idx}: {label_t} vs {label_s}")
        image_name = None
        if 'image' in self.data_csv.columns:
            image_name = self.data_csv.iloc[idx]['image']
        elif 'patient' in self.data_csv.columns:
            image_name = self.data_csv.iloc[idx]['patient']
        return img_t, img_s, label_s, image_name


class PairedFundusFullDataset(Dataset):
    """
    Returns (img_teacher, img_student, label, image_name) using full-image
    FundusClsDataset on both sides. This is useful when teacher and student
    share the same full-image input domain during KD.
    """

    def __init__(
        self,
        df,
        teacher_transform,
        student_transform,
        student_train=True,
        teacher_train=True,
    ):
        self.ds_teacher = FundusClsDataset(
            df,
            train=teacher_train,
            transform=teacher_transform,
        )
        self.ds_student = FundusClsDataset(
            df,
            train=student_train,
            transform=student_transform,
        )

        assert len(self.ds_teacher) == len(self.ds_student), "Teacher/Student datasets length mismatch."

        self.data_csv = df.reset_index(drop=True)

    def __len__(self):
        return len(self.ds_student)

    def __getitem__(self, idx):
        img_t, label_t = self.ds_teacher[idx]
        img_s, label_s = self.ds_student[idx]

        if label_t != label_s:
            raise RuntimeError(f"Label mismatch at idx {idx}: {label_t} vs {label_s}")

        image_name = None
        if "image" in self.data_csv.columns:
            image_name = self.data_csv.iloc[idx]["image"]
        elif "patient" in self.data_csv.columns:
            image_name = self.data_csv.iloc[idx]["patient"]

        return img_t, img_s, label_s, image_name


def _unique_patients_with_label(df, patient_col="patient_id", label_col="label"):
    patients = df[[patient_col, label_col]].drop_duplicates(subset=[patient_col]).reset_index(drop=True)
    return patients

def _can_stratify(patients_df, label_col="label"):
    counts = patients_df[label_col].value_counts()
    return (counts.min() >= 2) and (len(counts) >= 2)


def prepare_dataframes(
    cfg,
    patient_col: str = "patient_id",
    label_col: str = "label",
    val_rel: float = 0.15,
    test_rel: float = 0.15,
):
    data_csv = os.path.join(cfg.data_dir, cfg.csv_file)
    full_df = pd.read_csv(data_csv)

    if patient_col not in full_df.columns:
        full_df[patient_col] = (
            full_df["patient"].str.split("_").str[0].astype(str)
        )
    patients = _unique_patients_with_label(full_df, patient_col, label_col)

    if _can_stratify(patients, label_col):
        splitter_test = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_rel,
            random_state=cfg.RANDOM_SEED,
        )
        idx_trainval, idx_test = next(
            splitter_test.split(
                patients[[patient_col]],
                y=patients[label_col],
            )
        )
    else:
        splitter_test = GroupShuffleSplit(
            n_splits=1,
            test_size=test_rel,
            random_state=cfg.RANDOM_SEED,
        )
        idx_trainval, idx_test = next(
            splitter_test.split(
                patients,
                groups=patients[patient_col],
            )
        )

    trainval_patients = set(patients.iloc[idx_trainval][patient_col].values)
    test_patients     = set(patients.iloc[idx_test][patient_col].values)

    val_rel_eff = val_rel / (1.0 - test_rel)

    patients_trainval = patients[patients[patient_col].isin(trainval_patients)].reset_index(drop=True)

    if _can_stratify(patients_trainval, label_col):
        splitter_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_rel_eff,
            random_state=cfg.RANDOM_SEED,
        )
        idx_train, idx_val = next(
            splitter_val.split(
                patients_trainval[[patient_col]],
                y=patients_trainval[label_col],
            )
        )
    else:
        splitter_val = GroupShuffleSplit(
            n_splits=1,
            test_size=val_rel_eff,
            random_state=cfg.RANDOM_SEED,
        )
        idx_train, idx_val = next(
            splitter_val.split(
                patients_trainval,
                groups=patients_trainval[patient_col],
            )
        )

    train_patients = set(patients_trainval.iloc[idx_train][patient_col].values)
    val_patients   = set(patients_trainval.iloc[idx_val][patient_col].values)


    train_df = full_df[full_df[patient_col].isin(train_patients)].reset_index(drop=True)
    val_df   = full_df[full_df[patient_col].isin(val_patients)].reset_index(drop=True)
    test_df  = full_df[full_df[patient_col].isin(test_patients)].reset_index(drop=True)

    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)


    return train_df, val_df, test_df

