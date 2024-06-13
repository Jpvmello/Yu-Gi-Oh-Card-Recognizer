import os
import cv2
import glob
from torch.utils.data import Dataset


class CardsDataset(Dataset):
    def __init__(self, cards_dir, transform = None):
        self.cards_paths = glob.glob(os.path.join(cards_dir, '*.jpg'))
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