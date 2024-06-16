import argparse
from Model import Model
from CardsDataset import CardsDataset
import torchvision.transforms as T

from torch.utils.data import DataLoader, random_split


def parse_args():
    parser = argparse.ArgumentParser(
        description = "Yu-Gi-Oh! Card Recognizer: training a embedding of cards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_path", help = "Path or directory of the input card(s) image(s).")
    parser.add_argument("model_dir", help = "Directory to save the trained models.")
    parser.add_argument("-ep", "--epochs", default = 50, type = int, help = "Number of training epochs.")
    parser.add_argument("-vi", "--val-interval", default = 5, type = int, help = "Interval of epochs for performing validation.")
    parser.add_argument("-vp", "--val-prop", default = .2, type = float, help = "Proportion of data to be used for validation.")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    data = CardsDataset(args.input_dir, transform = T.Compose([T.ToTensor(), T.Resize((640, 448))])) # T.ToTensor() already normalizes
    train_data, val_data = random_split(data, [1 - args.val_prop, args.val_prop])
    
    train_dataloader = DataLoader(train_data, shuffle = True)
    val_dataloader   = DataLoader(val_data,   shuffle = False)

    model = Model()
    model.train(train_dataloader, val_dataloader, n_training_epochs = args.epochs,
        val_interval = args.val_interval, saved_models_dir = args.model_dir)





