import argparse
from matplotlib import pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as T
from tqdm.std import tqdm
from CardsDataset import CardsDataset
from Model import Model
import numpy as np
from sklearn.neighbors import NearestNeighbors


def parse_args():
    parser = argparse.ArgumentParser(
        description = "Yu-Gi-Oh! Card Recognizer: save the trained embedding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_path", help = "Path or directory of the input card(s) image(s).")
    parser.add_argument("embedding_path", help = "Path to the trained model.")
    parser.add_argument("output_path", help = "Path to save the embedding in .npy format.")
   
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    input_data = CardsDataset(args.input_path, transform = T.Compose([T.ToTensor(), T.Resize((640, 448))])) # T.ToTensor() already normalizes
    data = CardsDataset("cards_fullbody", transform = T.Compose([T.ToTensor(), T.Resize((640, 448))])) # T.ToTensor() already normalizes
    
    input_dataloader = DataLoader(input_data, shuffle = False)
    dataloader = DataLoader(data, shuffle = False)

    model = Model()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    with torch.no_grad():
        embedding = np.load(args.embedding_path)

        knn = NearestNeighbors(n_neighbors = 1, metric = "cosine")
        knn.fit(embedding)

        for image, _ in tqdm(input_dataloader):
            input  = np.moveaxis(image[0].cpu().detach().numpy(), 0, -1)
            encoding = model.encoder(image).cpu().detach().numpy()
            pred_idx = knn.kneighbors(encoding)

            prediction = np.moveaxis(
                dataloader.__getitem__(pred_idx)[0].cpu().detach().numpy(), 0, -1)

            plt.subplot(1, 2, 1)
            plt.title("Input")
            plt.imshow(input)
            plt.subplot(1, 2, 2)
            plt.title("Prediction")
            plt.imshow(prediction)
            plt.show()
            