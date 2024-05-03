import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt


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


# public methods are constructor, train, and load_model
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

        self.criterion = nn.MSELoss()

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

                logits = self.model(X).squeeze()
                logits = logits[:, -1]

                metric_true = (actual_after - actual_prev_day) / actual_prev_day
                metric_pred = (logits - actual_prev_day) / actual_prev_day

                loss = self.criterion(metric_true, metric_pred)

                self.optimizer.zero_grad()
                loss.backward()
                # TODO is this good?
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                epoch_loss += loss.item()

            self.scheduler.step()

            losses.append(epoch_loss)

            self.cur_epoch += 1

            if self.cur_epoch % 25 == 0:
                self.__save_model()

            average_loss = epoch_loss / len(dataloader.dataset)
            print(f"Epoch {self.cur_epoch}, Loss: {epoch_loss:.4f}; Average Loss: {average_loss}; lr: {self.scheduler.get_last_lr()}")

        self.model.eval()
        return self.__save_returns(losses)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        for (X, y) in dataloader:
            X = X.to(self.device)
            y = y.to(self.device)

            # y has 2 elements for each sample, as discussed in Datasets.py
            actual_after = y[:, 0]
            actual_prev_day = y[:, 1]

            logits = self.model(X).squeeze()
            logits = logits[:, -1]

            metric_true = (actual_after - actual_prev_day) / actual_prev_day
            metric_pred = (logits - actual_prev_day) / actual_prev_day

            # TODO print the actual calculated MSE for each row and the raw values used to do so (actualprevday, actual after, logits)
            # and print loss.item() to make sure pytorchs loss is just getting the average mse

            loss = self.criterion(metric_true, metric_pred)
            total_loss += loss.item()

        print(f"Average Loss: {total_loss / len(dataloader.dataset)}")




    # TODO maybe try doing labels for all encoder outputs, not just the last one?

    # TODO maybe normalize inputs and outputs to be 0 to 1? subtract mean??


def plot_losses(return_file):
    returns = np.load(return_file)

    plt.plot([i for i in range(len(returns))], returns)
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
