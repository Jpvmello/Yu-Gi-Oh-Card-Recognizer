import argparse
import os
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as T
from tqdm.std import tqdm
from CardsDataset import CardsDataset
from Model import Model
import pickle
from annoy import AnnoyIndex


def parse_args():
    parser = argparse.ArgumentParser(
        description = "Yu-Gi-Oh! Card Recognizer: save the trained embedding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_path", help = "Path or directory of the input card(s) image(s).")
    parser.add_argument("model_path", help = "Path to the trained model.")
    parser.add_argument("output_path", help = "Path to save the embedding in .npy format.")
    #parser.add_argument("--chunk-size", default = 500, type = int,
    #    help = "The embedding is saved on disk at each chunk_size images processed, for saving memory.")
   
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    output_extension = '.ann'
    output_path = args.output_path if args.output_path.endswith(output_extension) else (args.output_path + output_extension)
    dimension_output_path = os.path.splitext(output_path)[0] + '_dim.pkl'

    data = CardsDataset(args.input_path, transform = T.Compose([T.Resize((640, 448)), T.RandomEqualize(p = 1), T.ToTensor()])) # T.ToTensor() already normalizes
    dataloader = DataLoader(data, shuffle = False)

    model = Model()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    with torch.no_grad():
        embedding = None
        for i, (image, _) in enumerate(tqdm(dataloader)):
            encoding = model.encoder(image)[0].cpu().detach().numpy()
            if embedding is None:
                dimension = encoding.shape[0]
                embedding = AnnoyIndex(dimension, 'euclidean')

            embedding.add_item(i, encoding)

    print(f'Saving {output_path}...')
    embedding.build(n_trees = 10)
    embedding.save(output_path)
    with open(dimension_output_path, 'wb') as dim_out_file:
        pickle.dump(dimension, dim_out_file)
    print(output_path, "saved.")
            

