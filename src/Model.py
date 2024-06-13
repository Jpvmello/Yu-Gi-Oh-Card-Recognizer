import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from Encoder import Encoder
from Decoder import Decoder


class Model(nn.Module):
    def __init__(self, train_dataloader, val_dataloader, n_training_epochs, val_interval):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.loss_function = nn.MSELoss()

        self.train_dataloader = train_dataloader
        self.n_training_epochs = n_training_epochs
        self.val_interval = val_interval

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.optimizer = optim.Adam(self.parameters())

    def train_step(self):
        self.encoder.train()
        self.decoder.train()

        for input_image, target_image in tqdm(self.train_dataloader):
            input_image  = input_image.to(self.device)
            target_image = target_image.to(self.device)

            self.optimizer.zero_grad()

            features = self.encoder(input_image)
            output_image = self.decoder(features)

            loss = self.loss_function(output_image, target_image)
            loss.backward()

            self.optimizer.step()

        print('Training loss:', loss.item()) # reduction = 'mean' by default

    def val_step(self):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            for input_image, target_image in tqdm(self.val_dataloader):
                input_image  = input_image.to(self.device)
                target_image = target_image.to(self.device)

                features = self.encoder(input_image)
                output_image = self.decoder(features)

                loss = self.loss_function(output_image, target_image)

        print('Validation loss:', loss.item())

    def train(self):
        for epoch in range(1, self.n_training_epochs + 1):
            print(f'Epoch {epoch}/{self.n_training_epochs}')

            self.train_step()

            if epoch % self.val_interval == 0:
                self.val_step()