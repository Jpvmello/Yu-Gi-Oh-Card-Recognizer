import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from Encoder import Encoder
from Decoder import Decoder


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.loss_function = nn.MSELoss()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.optimizer = optim.Adam(self.parameters())

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def train_step(self, train_dataloader):
        self.encoder.train()
        self.decoder.train()

        for input_image, target_image in tqdm(train_dataloader):
            input_image  = input_image.to(self.device)
            target_image = target_image.to(self.device)

            self.optimizer.zero_grad()

            features = self.encoder(input_image)
            output_image = self.decoder(features)

            loss = self.loss_function(output_image, target_image)
            loss.backward()

            self.optimizer.step()

        print('Training loss:', loss.item()) # reduction = 'mean' by default

    def val_step(self, val_dataloader):
        self.eval()

        with torch.no_grad():
            for input_image, target_image in tqdm(val_dataloader):
                input_image  = input_image.to(self.device)
                target_image = target_image.to(self.device)

                features = self.encoder(input_image)
                output_image = self.decoder(features)

                loss = self.loss_function(output_image, target_image)

        val_loss = loss.item()
        print('Validation loss:', val_loss)
        return val_loss

    def train(self, train_dataloader, val_dataloader, n_training_epochs, val_interval, saved_models_dir):
        os.makedirs(saved_models_dir, exist_ok = True)
        
        best_val_loss = float('inf')
        for epoch in range(1, n_training_epochs + 1):
            print(f'Epoch {epoch}/{n_training_epochs}')

            self.train_step(train_dataloader)

            if epoch % val_interval == 0:
                val_loss = self.val_step(val_dataloader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.state_dict(), os.path.join(saved_models_dir, f"best_model_ep{epoch}.pth"))
    
    def forward(self, x):
        return self.decoder.forward(
            self.encoder.forward(x)
        )