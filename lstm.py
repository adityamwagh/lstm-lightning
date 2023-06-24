import lightning as L
import torch
import torch.nn as nn
from torch.optim import Adam


class LSTMbyHand(L.LightningModule):
    def __init__(self):
        """
        Create and initialize Weight and Bias tensors
        """
        super().__init__()

        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        # lr = % long term to remember
        self.wlr1 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.blr1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # pr = % potential memory to remember
        self.wpr1 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # p = potential long term memory
        self.wp1 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # o = output cell
        self.wo1 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean, std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def lstm_unit(self, input_value, long_memory, short_memory):
        """
        Do the LSTM math
        """

        # first stage
        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) + (input_value * self.wlr2) + self.blr1)

        # second stage
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) + (input_value * self.wpr2) + self.bpr1)
        potential_memory = torch.sigmoid((short_memory * self.wp1) + (input_value * self.wp2) + self.bp1)
        updated_long_memory = (long_memory * long_remember_percent) + (potential_memory * potential_remember_percent)

        # third stage
        output_percent = torch.sigmoid((short_memory * self.wo1) + (input_value * self.wo2) + self.bo1)
        updated_short_memory = torch.tanh(updated_long_memory) * output_percent

        return [updated_long_memory, updated_short_memory]

    def forward(self, input):
        """
        Make a forward pass through the unroled LSTM
        """
        long_memory = 0
        short_memory = 0

        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]

        long_memory, short_memory = self.lstm_unit(day1, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day2, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day3, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day4, long_memory, short_memory)

        return short_memory

    def configure_optimizers(self):
        """
        Configure Adam optimizer
        """
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        """
        Calculate loss and log training progress
        """
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i) ** 2

        self.log("train_loss", loss)  # creates lightning_log directory to store logging data

        if label_i == 0:
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)


class BuiltInLSTM(L.LightningModule):
    def __init__(self):
        """
        Create and initialize Weight and Bias tensors
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=1)

    def forward(self, input):
        """
        Make a forward pass through the unroled LSTM
        """
        input_trans = input.view(len(input), 1)
        lstm_out, _ = self.lstm(input_trans)

        prediction = lstm_out[-1]
        return prediction

    def configure_optimizers(self):
        """
        Configure Adam optimizer
        """
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        """
        Calculate loss and log training progress
        """
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i) ** 2

        self.log("train_loss", loss)  # creates lightning_log directory to store logging data

        if label_i == 0:
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)
