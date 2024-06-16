import argparse
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as T
from tqdm.std import tqdm
from CardsDataset import CardsDataset
from Model import Model
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description = "Yu-Gi-Oh! Card Recognizer: save the trained embedding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_path", help = "Path or directory of the input card(s) image(s).")
    parser.add_argument("model_path", help = "Path to the trained model.")
    parser.add_argument("output_path", help = "Path to save the embedding in .npy format.")
   
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    data = CardsDataset(args.input_path, transform = T.Compose([T.ToTensor(), T.Resize((640, 448))])) # T.ToTensor() already normalizes
    dataloader = DataLoader(data, shuffle = False)

    model = Model()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    with torch.no_grad():
        embedding = None
        for image, _ in tqdm(dataloader):
            encoding = model.encoder(image).cpu().detach().numpy()
            if embedding is None:
                embedding = encoding
            else:
                embedding = np.concatenate((embedding, encoding), axis = 0)
    
    output_path = args.output_path
    if output_path.endswith(".npy"):
        output_path += ".npy"
    np.save(output_path, embedding)
    print(output_path, "saved.")
            

