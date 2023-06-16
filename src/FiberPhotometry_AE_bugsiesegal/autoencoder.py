import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Compression(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(Compression, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


class Encoder(Compression):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(Encoder, self).__init__(input_size, hidden_size, num_layers=num_layers)

        for i in range(num_layers):
            self.add_module(f'linear{i}', nn.Linear(input_size, hidden_size))
            self.add_module(f'relu{i}', nn.ReLU())
            input_size = hidden_size

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.__getattr__(f'linear{i}')(x)
            x = self.__getattr__(f'relu{i}')(x)
        return x


class Decoder(Compression):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(Decoder, self).__init__(input_size, hidden_size, num_layers=num_layers)

        for i in range(num_layers):
            self.add_module(f'linear{i}', nn.Linear(input_size, hidden_size))
            self.add_module(f'relu{i}', nn.ReLU())
            input_size = hidden_size

        # self.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.__getattr__(f'linear{i}')(x)
            x = self.__getattr__(f'relu{i}')(x)
        # x = self.__getattr__('sigmoid')(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers=num_layers)
        self.decoder = Decoder(hidden_size, input_size, num_layers=num_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, data: Dataset, epochs=10, batch_size=32, learning_rate=1e-3, verbose=False,
            val_split=0.2, device='cpu', accumulation_steps=4, patience=5, factor=0.9):

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
        criterion = nn.MSELoss()

        train_idx, val_idx = train_test_split(list(range(len(data))), test_size=val_split)
        datasets = {'train': Subset(data, train_idx), 'val': Subset(data, val_idx)}

        train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(datasets['val'], batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')

        self.to(device)

        pbar = tqdm(total=epochs, dynamic_ncols=True)

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0

            for i, inputs in enumerate(train_loader):
                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()

                if (i + 1) % accumulation_steps == 0:  # Wait for several backward steps
                    optimizer.step()  # Now we can do an optimizer step
                    optimizer.zero_grad()  # Reset gradients tensors

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)

            val_loss = 0.0
            self.eval()
            with torch.no_grad():
                for inputs in val_loader:
                    outputs = self(inputs)
                    loss = criterion(outputs, inputs)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            if verbose:
                print(f'Epoch: {epoch + 1} | Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}')

            # Get current learning rate via the optimizer.
            current_lr = optimizer.param_groups[0]['lr']

            pbar.set_description(
                f'Epoch: {epoch + 1} | Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}')
            pbar.update()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = self.state_dict()

            # Step the learning rate scheduler
            scheduler.step(avg_val_loss)

        self.load_state_dict(best_model)
        pbar.close()
        return self

    def predict(self, data: Dataset, batch_size=32):
        self.eval()
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        predictions = []
        with torch.no_grad():
            for inputs in loader:
                outputs = self(inputs)
                predictions.append(outputs)
        return torch.cat(predictions, dim=0)


