import json
import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset


class COCODataset(Dataset):  # type: ignore
    """
    Dataset class for COCO-style object detection.

    This class is designed to load an object detection dataset in the COCO format. It expects
    a directory with images and a JSON file with annotations. The dataset returns a sample
    comprising an image, its bounding boxes, and the corresponding labels.

    Attributes:
        image_dir (str): Directory containing all the images.
        annotations (Dict[str, Any]): Loaded COCO annotation file's content.
        images (List[Dict[str, Any]]): List of image metadata from COCO annotations.
        categories (Dict[int, str]): Mapping of category IDs to category names.
        transform (Optional[Callable]): A function/transform that takes in a sample and returns a transformed version.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """
        Initializes the COCODataset instance.

        Args:
            image_dir (str): Directory with all the images.
            annotation_file (str): Path to the JSON file with annotations.
            transform (Callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform

        with open(annotation_file) as f:
            self.annotations = json.load(f)

        self.images = self.annotations["images"]
        self.categories = {category["id"]: category["name"] for category in self.annotations["categories"]}

    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves the item (image and its annotations) at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the image, bounding boxes, and labels.
        """
        img_metadata = self.images[idx]
        img_id = img_metadata["id"]
        img_path = os.path.join(self.image_dir, img_metadata["file_name"])
        image = Image.open(img_path).convert("RGB")

        annotations = [anno for anno in self.annotations["annotations"] if anno["image_id"] == img_id]
        boxes = [anno["bbox"] for anno in annotations]
        labels = [anno["category_id"] for anno in annotations]

        sample = {
            "image": image,
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
