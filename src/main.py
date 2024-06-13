from Model import Model
from CardsDataset import CardsDataset
import torchvision.transforms as T

from torch.utils.data import DataLoader, random_split


if __name__ == '__main__':
    data = CardsDataset("cards_fullbody", transform = T.Compose([T.ToTensor(), T.Resize((640, 448))])) # T.ToTensor() already normalizes
    train_data, val_data = random_split(data, [.8, .2])
    
    train_dataloader = DataLoader(train_data, shuffle = True)
    val_dataloader   = DataLoader(val_data,   shuffle = False)

    model = Model(train_dataloader, val_dataloader, n_training_epochs = 4, val_interval = 2)
    model.train()





