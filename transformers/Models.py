import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader
import math
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from Datasets import IncrementalDataset


# no embedding because we are using numerical input values (that were initially numbers, not converted to numbers)
# uses a default square triangular causal mask
class EncoderOnlyTransformer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_atn_heads, num_layers, dropout=0.1):
        super().__init__()

        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_atn_heads, hidden_dim, dropout)
        self.transformer = TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=False)

        self.linear = nn.Linear(embedding_dim, 1)
        self.init_weights()

    def init_weights(self):
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-.1, .1)

    def forward(self, x):
        # x is of dim [batch, #tokens, embedding_dim]
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(len(x)).to(x.device)

        logits = self.transformer(x, causal_mask)
        logits = logits.permute(1, 0, 2)

        # returns [batch, #tokens, 1]
        return self.linear(logits)


# from pytorch tutorial
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# public methods are constructor, train, evaluate, and load_model
class TransformerAgent:
    def __init__(self, embedding_dim, hidden_dim, num_atn_heads, num_layers, device, checkpoint_dir, init_lr, lr_decay, min_lr, decay_lr_every, dropout=0.1):
        self.checkpoint_dir = checkpoint_dir

        self.allow_training = True
        self.cur_epoch = 0

        self.device = device
        self.model = EncoderOnlyTransformer(embedding_dim, hidden_dim, num_atn_heads, num_layers, dropout)
        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=init_lr)

        # at epoch e, lr will be init_lr * scheduler_lamb(e)
        scheduler_lamb = lambda epoch: max(lr_decay ** (epoch // decay_lr_every), min_lr / init_lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scheduler_lamb)

        self.criterion = nn.MSELoss(reduction="sum")

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)

        self.model.load_state_dict(checkpoint["model"])
        self.cur_epoch = checkpoint["epoch"]

        self.allow_training = False
        self.model.eval()

    def __save_model(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        dt = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        path = os.path.join(self.checkpoint_dir, f"{dt}_agent_{self.cur_epoch}.pth")

        params = {
            "model": self.model.state_dict(),
            "epoch": self.cur_epoch
        }

        torch.save(params, path)

    def __save_returns(self, returns):
        returns_prefix = f"{self.checkpoint_dir}/returns"

        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if not os.path.isdir(returns_prefix):
            os.mkdir(returns_prefix)

        np_filename = f"{returns_prefix}/{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}.npy"
        np.save(np_filename, returns)

        return np_filename

    def train(self, dataloader, epochs):
        if not self.allow_training:
            raise Exception("Not allowed to train after loading")

        self.model.train()
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for (X, y) in dataloader:
                X = X.to(self.device)
                y = y.to(self.device)

                # y has 2 elements for each sample, as discussed in Datasets.py
                actual_after = y[:, 0]
                actual_prev_day = y[:, 1]

                logits = self.model(X).squeeze(-1)  # -1 in case batch size ends up being 1
                logits = logits[:, -1]

                metric_true = (actual_after - actual_prev_day) / actual_prev_day
                metric_pred = (logits - actual_prev_day) / actual_prev_day

                loss = self.criterion(metric_true, metric_pred)

                self.optimizer.zero_grad()
                loss.backward()
                # TODO is this good?
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                epoch_loss += loss.item()

            self.scheduler.step()

            losses.append(epoch_loss)

            self.cur_epoch += 1

            if self.cur_epoch % 25 == 0:
                self.__save_model()

            average_loss = epoch_loss / len(dataloader.dataset)  # epoch loss is the sum of each sample's loss since mse reduction is sum
            print(f"Epoch {self.cur_epoch}, Loss: {epoch_loss:.4f}; Average Loss: {average_loss}; lr: {self.scheduler.get_last_lr()}")

        self.model.eval()
        return self.__save_returns(losses)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(self.device)
            y = y.to(self.device)

            # y has 2 elements for each sample, as discussed in Datasets.py
            actual_after = y[:, 0]
            actual_prev_day = y[:, 1]

            logits = self.model(X).squeeze(-1)  # -1 in case batch size ends up being 1
            logits = logits[:, -1]

            metric_true = (actual_after - actual_prev_day) / actual_prev_day
            metric_pred = (logits - actual_prev_day) / actual_prev_day

            loss = self.criterion(metric_true, metric_pred)
            total_loss += loss.item()

            manual_mse = torch.pow(metric_pred - metric_true, 2)

            if batch % 100 == 0:
                print(f"!!!!!!!!!!!!!!!!\nBatch {batch}\n!!!!!!!!!!!!!!!\nactual prev day: {actual_prev_day}\n\n"
                      f"actual after: {actual_after}\n\npredicted after: {logits}\n\nmse: {manual_mse}\n\n"
                      f"pytorch loss: {loss.item()}")

        print(f"Average Loss: {total_loss / len(dataloader.dataset)}")  # total loss is the sum of each sample's loss since mse reduction is sum




    # TODO maybe try doing labels for all encoder outputs, not just the last one?

    # TODO maybe normalize inputs and outputs to be 0 to 1? subtract mean??


# Positional Encoding for Transformer (batch first)
class PositionalEncodingBF(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncodingBF, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class BigTransformer(nn.Module):
    def __init__(self, indim, outdim, hidden_dim=256, d_model=64, nhead=4, num_encoder_layers=6, num_decoder_layers=6):
        super(BigTransformer, self).__init__()

        self.pos_encoder = PositionalEncodingBF(d_model)
        self.embedding = nn.Sequential(
            nn.Linear(indim, hidden_dim),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
            )
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, batch_first=True)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, outdim)
        )

        self.d_model = d_model

    # src: (B, S, F) batches, sequence length, features
    # tgt: (B, F)
    def forward(self, src, tgt):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        tgt = self.embedding(tgt.unsqueeze(1))
        tgt = self.pos_encoder(tgt)
        src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
        out = self.transformer(src, tgt, src_mask, tgt_mask, src_is_causal=True, tgt_is_causal=True)
        out = self.decoder_mlp(out)
        return out.squeeze(1)
    
    def generate(self, src, max_len=100, start_token=None, end_token=None):
        device = src.device
        src = self.embedding(src)
        src = self.pos_encoder(src)
        src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(device)

        output_seq = torch.zeros((src.shape[0], 1, self.d_model)).to(device)
        if start_token is not None:
            output_seq[:, 0] = self.embedding(start_token.to(device))

        for i in range(max_len):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(output_seq.shape[1]).to(device)
            next_token = self.transformer(src, output_seq, src_mask=src_mask, tgt_mask=tgt_mask)
            output_seq = torch.cat([output_seq, next_token], dim=1)

            if end_token is not None and torch.all(next_token == end_token):
                break
            
        output_seq = self.decoder_mlp(output_seq[:, 1:]) # exclude start token
        
        return output_seq.squeeze(1)


# public methods are constructor, train, evaluate, and load_model
class BigTransformerAgent:
    def __init__(self, indim, outdim, hidden_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, 
                 device, checkpoint_dir, init_lr, lr_decay, min_lr, decay_lr_every):
        self.checkpoint_dir = checkpoint_dir

        self.allow_training = True
        self.cur_epoch = 0

        self.device = device
        self.model = BigTransformer(indim, outdim, hidden_dim, d_model, nhead, 
                                    num_encoder_layers, num_decoder_layers)
        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=init_lr)

        # at epoch e, lr will be init_lr * scheduler_lamb(e)
        scheduler_lamb = lambda epoch: max(lr_decay ** (epoch // decay_lr_every), min_lr / init_lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scheduler_lamb)

        self.criterion = nn.MSELoss(reduction='sum')

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)

        self.model.load_state_dict(checkpoint["model"])
        self.cur_epoch = checkpoint["epoch"]

        self.allow_training = False
        self.model.eval()

    def __save_model(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        dt = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        path = os.path.join(self.checkpoint_dir, f"{dt}_agent_{self.cur_epoch}.pth")

        params = {
            "model": self.model.state_dict(),
            "epoch": self.cur_epoch
        }

        torch.save(params, path)

    def __save_returns(self, returns):
        returns_prefix = f"{self.checkpoint_dir}/returns"

        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if not os.path.isdir(returns_prefix):
            os.mkdir(returns_prefix)

        np_filename = f"{returns_prefix}/{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}.npy"
        np.save(np_filename, returns)

        return np_filename

    def train(self, dataloader, epochs):
        if not self.allow_training:
            raise Exception("Not allowed to train after loading")

        self.model.train()
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for (X, y, tickers) in dataloader:
                X = X.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(X, y)

                loss = self.criterion(y, y_pred)

                self.optimizer.zero_grad()
                loss.backward()
                # TODO is this good?
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                epoch_loss += loss.item()

            # self.scheduler.step()

            losses.append(epoch_loss)

            self.cur_epoch += 1

            if self.cur_epoch % 25 == 0:
                self.__save_model()

            average_loss = epoch_loss / len(dataloader.dataset)  # epoch loss is the sum of each sample's loss since mse reduction is sum
            print(f"Epoch {self.cur_epoch}, Loss: {epoch_loss:.4f}; Average Loss: {average_loss}; lr: {self.scheduler.get_last_lr()}")

        self.model.eval()
        return self.__save_returns(losses)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        gt = torch.Tensor([])
        preds = torch.Tensor([])
        companies = []
        for batch, (X, y, co) in enumerate(dataloader):
            X = X.to(self.device)
            y = y.to(self.device)

            y_pred = self.model.generate(X, max_len=1)

            loss = self.criterion(y, y_pred)

            total_loss += loss.item()

            gt = torch.concat([gt, y], dim=0)
            preds = torch.concat([preds, y_pred], dim=0)
            companies.extend(co)

        print(f"Average Loss: {total_loss / len(dataloader.dataset)}")  # total loss is the sum of each sample's loss since mse reduction is sum

        gt = gt.to('cpu')
        preds = preds.to('cpu')

        return gt, preds, companies


# public methods are constructor, train, evaluate, and load_model
# agent for incremental learning models
class IncrementalAgent:
    def __init__(self, indim, outdim, hidden_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, 
                 device, checkpoint_dir, init_lr, lr_decay, min_lr, decay_lr_every, tickers):
        self.checkpoint_dir = checkpoint_dir
        self.tickers = tickers

        self.allow_training = True
        self.cur_epoch = 0

        self.device = device
        self.model = BigTransformer(indim, outdim, hidden_dim, d_model, nhead, 
                                    num_encoder_layers, num_decoder_layers)
        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=init_lr)

        # at epoch e, lr will be init_lr * scheduler_lamb(e)
        scheduler_lamb = lambda epoch: max(lr_decay ** (epoch // decay_lr_every), min_lr / init_lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scheduler_lamb)

        self.criterion = nn.MSELoss(reduction='sum')

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)

        self.model.load_state_dict(checkpoint["model"])
        self.cur_epoch = checkpoint["epoch"]

        self.allow_training = False
        self.model.eval()

    def __save_model(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        dt = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        path = os.path.join(self.checkpoint_dir, f"{dt}_agent_{self.cur_epoch}.pth")

        params = {
            "model": self.model.state_dict(),
            "epoch": self.cur_epoch
        }

        torch.save(params, path)

    def __save_returns(self, fn, returns):
        returns_prefix = f"{self.checkpoint_dir}/returns"

        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if not os.path.isdir(returns_prefix):
            os.mkdir(returns_prefix)

        np_filename = f"{returns_prefix}/{fn}{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}.npy"
        np.save(np_filename, returns)

        return np_filename


    def train(self, test_days, epochs):
        if not self.allow_training:
            raise Exception("Not allowed to train after loading")
        

        losses = []
        predictions = []
        ground_truths = []
        companies = []

        for day in range(test_days):
            # create new datasets for each new test_day
            train_set = IncrementalDataset(self.tickers, '../data_combined', 30, test_days - day, False)
            test_set = IncrementalDataset(self.tickers, '../data_combined', 30, test_days - day, True)
            train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
            test_loader = DataLoader(test_set)

            losses_per_day = []
            self.model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for (X, y, tickers) in train_loader:
                    X = X.to(self.device)
                    y = y.to(self.device)

                    y_pred = self.model(X, y)

                    loss = self.criterion(y_pred, y)

                    self.optimizer.zero_grad()
                    loss.backward()
                    # TODO is this good?
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

                    epoch_loss += loss.item()

                # self.scheduler.step()

                losses_per_day.append(epoch_loss)

                self.cur_epoch += 1

                if self.cur_epoch % 25 == 0:
                    self.__save_model()

                average_loss = epoch_loss / len(train_set)  # epoch loss is the sum of each sample's loss since mse reduction is sum
                print(f"Test Day {day}, Epoch {self.cur_epoch}, Loss: {epoch_loss:.4f}; Average Loss: {average_loss}; lr: {self.scheduler.get_last_lr()}")

            gt, pred, tickers = self.evaluate(test_loader)
            ground_truths.append(gt)
            predictions.append(pred)
            companies.extend(tickers)

            losses.append(losses_per_day)

        ground_truths = np.array(ground_truths)
        predictions = np.array(ground_truths)

        self.__save_returns('ground_truths_', ground_truths)
        self.__save_returns('predictions', predictions)

        self.model.eval()
        return ground_truths, predictions, companies

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        gt = torch.Tensor([])
        preds = torch.Tensor([])
        companies = []
        for batch, (X, y, co) in enumerate(dataloader):
            X = X.to(self.device)
            y = y.to(self.device)

            y_pred = self.model.generate(X, max_len=1)

            loss = self.criterion(y_pred, y)

            total_loss += loss.item()

            gt = torch.concat([gt, y], dim=0)
            preds = torch.concat([preds, y_pred], dim=0)
            companies.extend(co)

        print(f"Test Loss: {total_loss / len(dataloader.dataset)}")  # total loss is the sum of each sample's loss since mse reduction is sum

        gt = gt.to('cpu').detach().numpy()
        preds = preds.to('cpu').detach().numpy()

        return gt, preds, companies
    
# public methods are constructor, train, evaluate, and load_model
# agent for incremental learning models
class IncrementalProbAgent:
    def __init__(self, indim, outdim, hidden_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, 
                 device, checkpoint_dir, init_lr, lr_decay, min_lr, decay_lr_every, tickers):
        self.checkpoint_dir = checkpoint_dir
        self.tickers = tickers

        self.allow_training = True
        self.cur_epoch = 0

        self.device = device
        self.model = BigTransformer(indim, outdim, hidden_dim, d_model, nhead, 
                                    num_encoder_layers, num_decoder_layers)
        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=init_lr)

        # at epoch e, lr will be init_lr * scheduler_lamb(e)
        scheduler_lamb = lambda epoch: max(lr_decay ** (epoch // decay_lr_every), min_lr / init_lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scheduler_lamb)

        weight = torch.tensor([0.75]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=weight)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)

        self.model.load_state_dict(checkpoint["model"])
        self.cur_epoch = checkpoint["epoch"]

        self.allow_training = False
        self.model.eval()

    def __save_model(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        dt = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        path = os.path.join(self.checkpoint_dir, f"{dt}_agent_{self.cur_epoch}.pth")

        params = {
            "model": self.model.state_dict(),
            "epoch": self.cur_epoch
        }

        torch.save(params, path)

    def __save_returns(self, fn, returns):
        returns_prefix = f"{self.checkpoint_dir}/returns"

        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if not os.path.isdir(returns_prefix):
            os.mkdir(returns_prefix)

        np_filename = f"{returns_prefix}/{fn}{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}.npy"
        np.save(np_filename, returns)

        return np_filename

    def train(self, test_days, epochs):
        if not self.allow_training:
            raise Exception("Not allowed to train after loading")
        
        losses = []
        predictions = []
        ground_truths = []
        companies = []

        for day in range(test_days):
            # create new datasets for each new test_day
            train_set = IncrementalDataset(self.tickers, '../data_combined', 30, test_days - day, False)
            test_set = IncrementalDataset(self.tickers, '../data_combined', 30, test_days - day, True)
            train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_set)

            losses_per_day = []
            self.model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for (X, y, tickers) in train_loader:
                    X = X.to(self.device)
                    y = y.to(self.device)

                    y_pred = self.model(X, y) # y_pred will be (B, 1)

                    y = y[:,7].clone() # pct_change is 7th column
                    y[y <= 0] = 0
                    y[y > 0] = 1

                    loss = self.criterion(y_pred, y.unsqueeze(1))

                    self.optimizer.zero_grad()
                    loss.backward()
                    # TODO is this good?
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

                    epoch_loss += loss.item()

                # self.scheduler.step()

                losses_per_day.append(epoch_loss)

                self.cur_epoch += 1

                if self.cur_epoch % 25 == 0:
                    self.__save_model()

                average_loss = epoch_loss / len(train_set)  # epoch loss is the sum of each sample's loss since mse reduction is sum
                print(f"Test Day {day}, Epoch {self.cur_epoch}, Loss: {epoch_loss:.4f}; Average Loss: {average_loss}; lr: {self.scheduler.get_last_lr()}")

            gt, pred, tickers = self.evaluate(test_loader)
            ground_truths.append(gt)
            predictions.append(pred)
            companies.extend(tickers)

            losses.append(losses_per_day)

        ground_truths = np.array(ground_truths).squeeze()
        predictions = np.array(predictions).squeeze()

        self.__save_returns('ground_truths_', ground_truths)
        self.__save_returns('predictions', predictions)

        self.model.eval()
        return ground_truths, predictions, companies

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        gt = torch.Tensor([]).to(self.device)
        preds = torch.Tensor([]).to(self.device)
        companies = []
        for batch, (X, y, co) in enumerate(dataloader):
            X = X.to(self.device)
            y = y.to(self.device)

            y_pred = self.model.generate(X, max_len=1)

            y = y[:,7].clone() # pct_change is 7th column
            y[y <= 0] = 0
            y[y > 0] = 1

            loss = self.criterion(y_pred, y.unsqueeze(1))

            total_loss += loss.item()

            gt = torch.concat([gt, y.unsqueeze(1)], dim=0)
            preds = torch.concat([preds, torch.sigmoid(y_pred)], dim=0)
            companies.extend(co)

        print(f"Test Loss: {total_loss / len(dataloader.dataset)}")  # total loss is the sum of each sample's loss since mse reduction is sum

        gt = gt.to('cpu').detach().numpy()
        preds = preds.to('cpu').detach().numpy()

        return gt, preds, companies

def plot_losses(return_file):
    returns = np.load(return_file)

    plt.plot([i for i in range(len(returns))], returns)
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
