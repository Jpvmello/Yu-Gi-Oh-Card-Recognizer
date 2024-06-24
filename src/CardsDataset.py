import os
from PIL import Image
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
        image = Image.open(self.cards_paths[idx])
        
        if self.transform is not None:
            try:
                image = self.transform(image)
            except:
                print(idx, self.cards_paths[idx])
        
        return image, image