import torch
import torch.nn as nn
from ConvLSTMCell import ConvLSTMCell

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class ConvLSTM(nn.Module):

    # def __init__(self, in_channels, out_channels, 
    # kernel_size, padding, activation, frame_size):

    #     super(ConvLSTM, self).__init__()

    #     self.out_channels = out_channels

    #     # We will unroll this over time steps
    #     self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, 
    #     kernel_size, padding, activation, frame_size)

    # def forward(self, X):

    #     # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

    #     # Get the dimensions
    #     batch_size, _, seq_len, height, width = X.size()

    #     # Initialize output
    #     output = torch.zeros(batch_size, self.out_channels, seq_len, 
    #     height, width, device=device)
        
    #     # Initialize Hidden State
    #     H = torch.zeros(batch_size, self.out_channels, 
    #     height, width, device=device)

    #     # Initialize Cell Input
    #     C = torch.zeros(batch_size,self.out_channels, 
    #     height, width, device=device)

    #     # Unroll over time steps
    #     for time_step in range(seq_len):

    #         H, C = self.convLSTMcell(X[:,:,time_step], H, C)

    #         output[:,:,time_step] = H

    #     return output




import torch
from torch import nn
from typing import Optional, Union
# from . import CfCCell, WiredCfCCell
# from .lstm import LSTMCell

class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        units,
        proj_size: Optional[int] = None,
        return_sequences: bool = True,
        batch_first: bool = True,
        mixed_memory: bool = False,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: Optional[int] = None,
        backbone_layers: Optional[int] = None,
        backbone_dropout: Optional[int] = None,
    ):
        """Applies a `Closed-form Continuous-time <https://arxiv.org/abs/2106.13898>`_ RNN to an input sequence.

        Examples::

             >>> from ncps.torch import CfC
             >>>
             >>> rnn = CfC(20,50)
             >>> x = torch.randn(2, 3, 20) # (batch, time, features)
             >>> h0 = torch.zeros(2,50) # (batch, units)
             >>> output, hn = rnn(x,h0)

        :param input_size: Number of input features
        :param units: Number of hidden units
        :param proj_size: If not None, the output of the RNN will be projected to a tensor with dimension proj_size (i.e., an implict linear output layer)
        :param return_sequences: Whether to return the full sequence or just the last output
        :param batch_first: Whether the batch or time dimension is the first (0-th) dimension
        :param mixed_memory: Whether to augment the RNN by a `memory-cell <https://arxiv.org/abs/2006.04418>`_ to help learn long-term dependencies in the data
        :param mode: Either "default", "pure" (direct solution approximation), or "no_gate" (without second gate).
        :param activation: Activation function used in the backbone layers
        :param backbone_units: Number of hidden units in the backbone layer (default 128)
        :param backbone_layers: Number of backbone layers (default 1)
        :param backbone_dropout: Dropout rate in the backbone layers (default 0)
        """

        super(ConvLSTM, self).__init__()
        self.input_size = input_size
        self.proj_size = proj_size
        self.batch_first = batch_first
        self.return_sequences = return_sequences

        backbone_units = 128 if backbone_units is None else backbone_units
        backbone_layers = 1 if backbone_layers is None else backbone_layers
        backbone_dropout = 0.0 if backbone_dropout is None else backbone_dropout
        self.state_size = units
        self.output_size = self.state_size


        # print(input_size[:2] + input_size[3:])
        self.rnn_cell = ConvLSTMCell(
            input_size  = input_size[:2] + input_size[3:],
            hidden_size = units,
            backbone_units  = backbone_units,
            backbone_layers = backbone_layers
        )

        self.use_mixed = mixed_memory
        if self.use_mixed:
            self.lstm = None

        if proj_size is None:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.output_size, self.proj_size)

    def forward(self, input, hx=None, timespans=None):
        """

        :param input: Input tensor of shape (L,C) in batchless mode, or (B,L,C) if batch_first was set to True and (L,B,C) if batch_first is False
        :param hx: Initial hidden state of the RNN of shape (B,H) if mixed_memory is False and a tuple ((B,H),(B,H)) if mixed_memory is True. If None, the hidden states are initialized with all zeros.
        :param timespans:
        :return: A pair (output, hx), where output and hx the final hidden state of the RNN
        """


        input = torch.permute(input, (0,2,1,3,4))


        device = input.device
        batch_size, seq_len = input.shape[0], input.shape[1] # (B,Seq.,C,H,W)

        if hx is None:
            h_state = torch.zeros(self.state_size, device=device)
        else:
            h_state = hx

        output_sequence = []
        for t in range(seq_len):
            if self.batch_first:
                inputs = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            else:
                inputs = input[t]
                ts = 1.0 if timespans is None else timespans[t].squeeze()
            h_state = self.rnn_cell.forward(inputs, h_state, ts)
            if self.return_sequences:
                output_sequence.append(self.fc(h_state))

        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = self.fc(h_state)

        hx = h_state

        return readout.permute(0,2,1,3,4)#, hx

        