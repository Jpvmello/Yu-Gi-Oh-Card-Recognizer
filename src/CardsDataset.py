import os
import cv2
import glob
from torch.utils.data import Dataset


class CardsDataset(Dataset):
    def __init__(self, cards_path, transform = None):
        path_is_dir = os.path.isdir(cards_path)

        self.cards_paths = glob.glob(os.path.join(cards_path, '*.jpg')) if path_is_dir else [cards_path]
        self.transform = transform

    def __len__(self):
        return len(self.cards_paths)
        
    def __getitem__(self, idx):
        image = cv2.cvtColor(
            cv2.imread(self.cards_paths[idx]),
            cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, image