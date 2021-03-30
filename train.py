# Standard Modules
import json
import os
import sys

# Other Modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
import tqdm

# Custom Modules


def main():
    checkpoint_callback = ModelCheckpoint(
        filename="best", monitor="val_loss", save_last=True
    )
    lr_monitor = LearningRateMonitor()
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=10)

    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        gpus=torch.cuda.device_count(),
        default_root_dir="checkpoints",
        max_epochs=20,
        callbacks=[
            checkpoint_callback,
            lr_monitor,
            early_stopping,
        ],
    )

    model = Model()

    mode = 1

    if mode == 0:
        # model = model.load_from_checkpoint(
        #    "checkpoints/lightning_logs/version_19/checkpoints/best.ckpt"
        # )
        trainer.fit(model)
        print(checkpoint_callback.best_model_path)
        model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
        trainer.test(model)
    elif mode == 1:
        model = model.load_from_checkpoint(
            "checkpoints/lightning_logs/version_19/checkpoints/best.ckpt"
        )

        if torch.cuda.device_count():
            model.cuda()

        model.prepare_data()

        while True:
            model.test_random_sample()

    elif mode == 2:
        model = model.load_from_checkpoint(
            # "checkpoints/lightning_logs/version_19/checkpoints/best.ckpt"
            "checkpoints/lightning_logs/version_19/checkpoints/last.ckpt"
        )
        trainer.test(model)

    return


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.batch_size = 64
        self.n_features = 4
        self.d_model = 512
        self.nhead = 8
        self.dim_feedforward = 2048
        self.dropout = 0.5
        self.activation = "gelu"
        self.num_layers = 8

        mask = self.generate_square_subsequent_mask(8 * 24 - 1)
        self.register_buffer("mask", mask, persistent=False)

        loss_weight = torch.pow(torch.arange(24, 0, step=-1, dtype=torch.float), 2)[
            :, None, None
        ]
        loss_weight = loss_weight / loss_weight.mean()
        self.register_buffer("loss_weight", loss_weight, persistent=False)

        self.fc1 = nn.Linear(self.n_features, self.d_model)

        self.pos_encoder = PositionalEncoding(self.d_model, 0.5)

        decoder_layer = TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer, num_layers=self.num_layers
        )

        self.fc2 = nn.Linear(self.d_model, self.n_features)

    def forward(self, src):
        src = self.fc1(src)
        src = self.pos_encoder(src)
        out = self.transformer_decoder(
            src,
            tgt_mask=self.mask,
        )
        out = self.fc2(out)
        return out

    def training_step(self, batch, batch_idx):
        src, tgt = batch

        # View with shape {sentence length, batch size, features}
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        pred = self(src)

        # Calculate loss of only last 24h
        pred = pred[-24:]
        tgt = tgt[-24:]

        # loss = F.mse_loss(pred, tgt)
        loss = weighted_mse_loss(pred, tgt, self.loss_weight)
        # self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        src, tgt = batch

        # View with shape {sentence length, batch size, features}
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # Mask last entries of the src sequence that should be predicted
        src[-24 + 1 :] = 0

        for i in range(-24, 0):
            pred = self(src)
            src[i + 1] = pred[i]

        # Calculate loss of only last 24h and using only High and Low values
        pred = pred[-24:, :, 1:3]
        tgt = tgt[-24:, :, 1:3]

        loss = weighted_mse_loss(pred, tgt, self.loss_weight)
        self.log("test_loss", loss)

        return loss

    def test_random_sample(self):
        self.eval()
        torch.set_grad_enabled(False)

        sample = self.dataset.test[torch.randint(len(self.dataset.test), size=(1,))]
        src, tgt = sample

        src = src.to(self.device)
        tgt = tgt.to(self.device)

        # View with shape {sentence length, batch size, features}
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # Mask last entries of the src sequence that should be predicted
        src[-24 + 1 :] = 0

        for i in tqdm.trange(-24, 0, desc="Predicting"):
            pred = self(src)
            if i < -1:
                src[i + 1] = pred[i]  # 1st entry of src is messed up in last iteration

        # Calculate loss of only last 24h and using only High and Low values
        pred = pred[-24:, :, 1:3]
        tgt = tgt[-24:, :, 1:3]

        loss = weighted_mse_loss(pred, tgt, self.loss_weight).cpu()
        print("test_loss", loss)

        history = src[: -24 + 1, :, 1:3].squeeze(1)
        pred = torch.cat([history[-1:, :], pred.squeeze(1)], dim=0)
        tgt = torch.cat([history[-1:, :], tgt.squeeze(1)], dim=0)

        time1 = range(len(history))
        time2 = range(len(history) - 1, len(history) - 1 + len(pred))
        history = history.cpu()
        pred = pred.cpu()
        tgt = tgt.cpu()

        plt.plot(time1, history[:, 0], label="history high", color="darkgreen")
        plt.plot(time1, history[:, 1], label="history low", color="darkred")
        plt.plot(time2, pred[:, 0], label="prediction high", color="limegreen")
        plt.plot(time2, pred[:, 1], label="prediction low", color="red")
        plt.plot(time2, tgt[:, 0], label="real high", color="limegreen", ls="--")
        plt.plot(time2, tgt[:, 1], label="real low", color="red", ls="--")

        plt.title("Bitcoin price prediction of 1 day from past 7 days")
        plt.xlabel("time")
        plt.ylabel("normalized value")
        plt.grid()
        plt.legend()

        plt.show()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        scheduler = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=1e-4,
                total_steps=30,
                div_factor=10,
                final_div_factor=10,
                verbose=True,
            ),
            "interval": "epoch",
        }
        return [optimizer], [scheduler]
        # return optimizer

    def prepare_data(self):
        self.dataset = CryptoDataset(
            filename="data/BTCUSD_1hr.csv",
            seq_len=8 * 24,  # 7 days of history and 1 day to predict
            avg_size=6,  # average 12h
            avg_stride=1,
            norm_len=7 * 24,  # normalize wrt the 7 days of history
            fraction=0.95,
        )

    def train_dataloader(self):
        loader = DataLoader(
            dataset=self.dataset.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=bool(torch.cuda.device_count()),
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            dataset=self.dataset.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=bool(torch.cuda.device_count()),
        )
        return loader

    def test_dataloader(self):
        return self.val_dataloader()

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(
            torch.full((sz, sz), float("-inf"), dtype=torch.float, device=self.device),
            diagonal=1,
        )
        return mask

    def on_epoch_start(self):
        print()  # so that the progress bar remains for each epoch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class CryptoData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx, :-1], self.data[idx, 1:]


class CryptoDataset:
    def __init__(
        self,
        filename,
        seq_len=8 * 24,  # 7 days of history and 1 day to predict
        step=1,
        avg_size=None,  # no average
        avg_stride=None,
        norm_len=7 * 24,  # normalize wrt the 7 days of history
        fraction=0.95,
    ):
        basename = os.path.splitext(os.path.basename(filename))[0]
        train_dataset = os.path.join("dataset", basename + "_train.json")
        test_dataset = os.path.join("dataset", basename + "_test.json")

        if os.path.isfile(train_dataset) and os.pathisfile(test_dataset):
            print("Found dataset files")
        else:
            print("Dataset files not found. Creating datasets")
            dataset = self.process_data(
                filename, seq_len, step, avg_size, avg_stride, norm_len
            )
            train, test = self.split_dataset(dataset, fraction, shuffle=True)
            # TODO: save train and test
            self.train = CryptoData(train)
            self.test = CryptoData(test)

    def process_data(self, filename, seq_len, step, avg_size, avg_stride, norm_len):

        cols = {
            # "Unix Timestamp": int,
            "Open": np.float32,
            "High": np.float32,
            "Low": np.float32,
            "Close": np.float32,
        }

        data = pd.read_csv(filename, skiprows=1, usecols=cols.keys(), dtype=cols)

        # Get data, from oldest to newest, as torch tensor
        data = torch.flip(torch.tensor(data.values, dtype=torch.float32), dims=[0])

        # Smooth data
        if avg_size is not None and avg_size > 1:
            data = data.unsqueeze(dim=0).transpose(1, 2)
            data = F.avg_pool1d(data, avg_size, avg_stride)
            data = data.transpose(1, 2).squeeze(dim=0)

        # Logarithm of data
        # data = torch.log(data + 1)

        # Construct samples of size seq_len. Dims {batch size, sequence size, features}
        dataset = torch.stack(
            [
                data[i : i + seq_len, :]
                for i in tqdm.trange(
                    0, data.size(0) - seq_len + 1, step, desc="Constructing samples"
                )
            ]
        )

        # Normalize with respect to norm_len samples
        if norm_len is None:
            norm_len = seq_len
        elif norm_len > 0:
            #
            mean = dataset[:, :norm_len, :].mean(dim=[1, 2])[:, None, None]
            std = dataset[:, :norm_len, :].std(dim=[1, 2])[:, None, None]
            std[std < 1e-3] = 1  # to avoid division by 0
            dataset = (dataset - mean) / std

        """
        # Plot samples
        while True:
            i = np.random.randint(0, len(dataset))
            plt.plot(dataset[i, :, 1], ".-", label="high 1")
            plt.plot(dataset[i, :, 2], ".--", label="low 1")
            plt.legend()
            plt.show()
        """

        return dataset

    @staticmethod
    def split_dataset(dataset, fraction, shuffle=False):
        # Shuffle
        if shuffle:
            idx = torch.randperm(dataset.size(0))
            dataset = dataset[idx].view(dataset.size())

        n = int(dataset.size(0) * fraction)
        return dataset[:n], dataset[n:]


class TransformerDecoder(nn.Module):
    """Similar to PyTorch TransformerDecoder but without memory"""

    from typing import Optional

    __constants__ = ["norm"]

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = self._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        output = tgt

        for mod in self.layers:
            output = mod(
                output,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

    @staticmethod
    def _get_clones(module, N):
        import copy

        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoderLayer(nn.Module):
    """Similar to PyTorch TransformerDecoderLayer but without memory"""

    from typing import Optional

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def weighted_mse_loss(output, target, weight):
    return (weight * (output - target) ** 2).mean()


if __name__ == "__main__":
    main()
