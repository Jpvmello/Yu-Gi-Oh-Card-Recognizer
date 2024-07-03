import argparse
import os
import pickle
from matplotlib import pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as T
from tqdm.std import tqdm
from CardsDataset import CardsDataset
from Model import Model
import numpy as np
from annoy import AnnoyIndex

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Yu-Gi-Oh! Card Recognizer: predict the card.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_path", help = "Path or directory of the input card(s) image(s).")
    parser.add_argument("dataset_dir", help = "Directory of the cards images database.")
    parser.add_argument("model_path", help = "Path to the trained model.")
    parser.add_argument("embedding_path", help = "Path to the saved cards embedding.")
   
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dimension_file_path = os.path.splitext(args.embedding_path)[0] + '_dim.pkl'

    input_data = CardsDataset(args.input_path, transform = T.Compose([T.Resize((640, 448)), T.RandomEqualize(p = 1), T.ToTensor()])) # T.ToTensor() already normalizes
    data = CardsDataset(args.dataset_dir, transform = T.Compose([T.Resize((640, 448)), T.RandomEqualize(p = 1), T.ToTensor()])) # T.ToTensor() already normalizes
    
    input_dataloader = DataLoader(input_data, shuffle = True)
    dataloader = DataLoader(data, shuffle = False)

    model = Model()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    with torch.no_grad():
        print(f'Loading {args.embedding_path}...')
        with open(dimension_file_path, 'rb') as dim_file:
            dimension = pickle.load(dim_file)
        embedding = AnnoyIndex(dimension, 'euclidean')
        embedding.load(args.embedding_path)
        print(args.embedding_path, 'loaded.')

        for image, _ in tqdm(input_dataloader):
            input  = np.moveaxis(image[0].cpu().detach().numpy(), 0, -1)
            encoding = model.encoder(image)[0].cpu().detach().numpy()
            pred_idx = embedding.get_nns_by_vector(encoding, 1)[0]
            print(pred_idx)

            prediction = np.moveaxis(
                data.__getitem__(pred_idx)[0].cpu().detach().numpy(), 0, -1)

            plt.subplot(1, 2, 1)
            plt.title("Input")
            plt.imshow(input)
            plt.subplot(1, 2, 2)
            plt.title("Prediction")
            plt.imshow(prediction)
            plt.show()
            