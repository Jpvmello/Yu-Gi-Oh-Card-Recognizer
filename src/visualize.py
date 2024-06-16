import argparse
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as T
from CardsDataset import CardsDataset
from Model import Model
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description = "Yu-Gi-Oh! Card Recognizer: visualize reconstruction results.\n"\
            + "The results are exibhited card by card, alternating according to the given interval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_path", help = "Path or directory of the input card(s) image(s).")
    parser.add_argument("model_path", help = "Path to the trained model.")
    parser.add_argument("-i", "--interval", default = 10, type = int, help = "Time interval between exibhitions (in seconds).")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    data = CardsDataset(args.input_path, transform = T.Compose([T.ToTensor(), T.Resize((640, 448))])) # T.ToTensor() already normalizes
    dataloader = DataLoader(data, shuffle = True)

    model = Model()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    with torch.no_grad():

        plt.ion()
        for image, _ in dataloader:
            input  = np.moveaxis(image[0].cpu().detach().numpy(), 0, -1)
            output = np.moveaxis(model(image)[0].cpu().detach().numpy(), 0, -1)

            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(input)
            plt.subplot(1, 2, 2)
            plt.title("Reconstructed")
            plt.imshow(output)
            plt.draw()
            plt.pause(args.interval)
            plt.close()
