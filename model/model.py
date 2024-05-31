import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from torchmetrics.functional import mean_absolute_percentage_error


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TimeSeriesPredictionModel(LightningModule):
    def __init__(
        self,
        num_inputs,
        num_channels,
        num_outputs,
        kernel_size,
        dropout,
        learning_rate,
        min_y,
        max_y,
    ):
        super(TimeSeriesPredictionModel, self).__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], num_outputs)  # 2 outputs for BTC/EUR and ETH/EUR
        self.learning_rate = learning_rate
        self.min_y = torch.tensor(min_y, dtype=torch.float32)
        self.max_y = torch.tensor(max_y, dtype=torch.float32)

    def forward(self, x_tcn):
        y = self.tcn(x_tcn)
        return self.linear(y[:, :, -1])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        y_hat_unscaled = self.unscale_y(y_hat)
        y_unscaled = self.unscale_y(y)
        mape = mean_absolute_percentage_error(y_hat_unscaled, y_unscaled)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_mape", mape, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        y_hat_unscaled = self.unscale_y(y_hat)
        y_unscaled = self.unscale_y(y)
        mape = mean_absolute_percentage_error(y_hat_unscaled, y_unscaled)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_mape", mape, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def unscale_y(self, y):
        return y * (self.max_y - self.min_y) + self.min_y
