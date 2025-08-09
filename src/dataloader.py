import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random

class DukeMTMCDataset(Dataset):
    """
    Custom PyTorch Dataset for DukeMTMC-reID.
    This dataloader provides triplets of (anchor, positive, negative) images.
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.all_imgs = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
        
        # Group images by person ID for triplet sampling
        self.person_to_images = defaultdict(list)
        for img_path in self.all_imgs:
            person_id = os.path.basename(img_path).split('_')[0]
            # Ignore distractors or non-person IDs
            if person_id.isdigit():
                self.person_to_images[int(person_id)].append(img_path)

        self.person_ids = list(self.person_to_images.keys())
        print(f"Dataset initialized with {len(self.all_imgs)} images from {len(self.person_ids)} identities.")

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, index):
        # Anchor image
        anchor_path = self.all_imgs[index]
        anchor_pid = int(os.path.basename(anchor_path).split('_')[0])
        
        # Select a positive image (same person, different image)
        positive_list = self.person_to_images[anchor_pid]
        positive_path = random.choice(positive_list)
        # Ensure it's not the same image as the anchor
        while positive_path == anchor_path and len(positive_list) > 1:
            positive_path = random.choice(positive_list)

        # Select a negative image (different person)
        negative_pid = random.choice(self.person_ids)
        while negative_pid == anchor_pid:
            negative_pid = random.choice(self.person_ids)
        negative_path = random.choice(self.person_to_images[negative_pid])
        
        # Load images
        anchor_img = Image.open(anchor_path).convert("RGB")
        positive_img = Image.open(positive_path).convert("RGB")
        negative_img = Image.open(negative_path).convert("RGB")

        # Apply transformations
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            
        return anchor_img, positive_img, negative_img
