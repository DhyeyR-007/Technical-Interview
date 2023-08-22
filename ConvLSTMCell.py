import torch
import torch.nn as nn
# from attention_augmented_conv import AugmentedConv


# Original ConvLSTM cell as proposed by Shi et al.
# class ConvLSTMCell(nn.Module):

#     def __init__(self, in_channels, out_channels, 
#     kernel_size, padding, activation, frame_size):

#         super(ConvLSTMCell, self).__init__()  

#         if activation == "tanh":
#             self.activation = torch.tanh 
#         elif activation == "relu":
#             self.activation = torch.relu
        
#         # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
#         self.conv = nn.Conv2d(
#             in_channels=in_channels + out_channels, 
#             out_channels=4 * out_channels, 
#             kernel_size=kernel_size,
#             padding=padding)    


#         # self.conv = AugmentedConv(in_channels=in_channels + out_channels, out_channels=4 * out_channels, kernel_size=kernel_size[0], dk=40, dv=4, Nh=4, relative=True, stride=1, shape=frame_size[0])
#         # print(self.conv.shape)

#         # Initialize weights for Hadamard Products
#         self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
#         self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
#         self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))


#         # self.W_pi = nn.Parameter(torch.Tensor(out_channels, *frame_size))
#         # self.W_pf = nn.Parameter(torch.Tensor(out_channels, *frame_size))
#         # self.W_po = nn.Parameter(torch.Tensor(out_channels, *frame_size))

#     def forward(self, X, H_prev, C_prev):

#         # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
#         conv_output = self.conv(torch.cat([X, H_prev], dim=1))

#         # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
#         i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

#         input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )
#         forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev )

#         # Current Cell output
#         C = forget_gate*C_prev + input_gate * self.activation(C_conv)

#         output_gate = torch.sigmoid(o_conv + self.W_co * C )

#         # Current Hidden State
#         H = output_gate * self.activation(C)

#         return H, C
    


### separate add x and cprev lstm
    # def forward(self, X, H_prev, C_prev):
    #         conv_output = self.conv(torch.cat([X, H_prev], dim=1))

    #         i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

    #         input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev + self.W_pi * X)
    #         forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev + self.W_pf * X)

    #         C = forget_gate * C_prev + input_gate * self.activation(C_conv)

    #         output_gate = torch.sigmoid(o_conv + self.W_co * C + self.W_po * X)

    #         H = output_gate * self.activation(C)

    #         return H, C





import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Union


class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        mode="default",
        backbone_activation="lecun_tanh",
        backbone_units=128,
        backbone_layers=3,
        backbone_dropout=0.0,
        sparsity_mask=None,
    ):
        """A `Closed-form Continuous-time <https://arxiv.org/abs/2106.13898>`_ cell.

        .. Note::
            This is an RNNCell that process single time-steps. To get a full RNN that can process sequences see `ncps.torch.CfC`.



        :param input_size:
        :param hidden_size:
        :param mode:
        :param backbone_activation:
        :param backbone_units:
        :param backbone_layers:
        :param backbone_dropout:
        :param sparsity_mask:
        """

        super(ConvLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(
                f"Unknown mode '{mode}', valid options are {str(allowed_modes)}"
            )
        self.sparsity_mask = (
            None
            if sparsity_mask is None
            else torch.nn.Parameter(
                data=torch.from_numpy(np.abs(sparsity_mask.T).astype(np.float32)),
                requires_grad=False,
            )
        )

        self.mode = mode

        if backbone_activation == "silu":
            backbone_activation = nn.SiLU
        elif backbone_activation == "relu":
            backbone_activation = nn.ReLU
        elif backbone_activation == "tanh":
            backbone_activation = nn.Tanh
        elif backbone_activation == "gelu":
            backbone_activation = nn.GELU
        elif backbone_activation == "lecun_tanh":
            backbone_activation = LeCun
        else:
            raise ValueError(f"Unknown activation {backbone_activation}")

        self.backbone = None
        self.backbone_layers = backbone_layers
        if backbone_layers > 0:
            layer_list = [
               # nn.Linear(input_size + hidden_size, backbone_units),
                nn.Conv2d(in_channels=input_size[1] + hidden_size[1], out_channels=backbone_units, kernel_size=3, padding=1),
                backbone_activation()
            ]
            for i in range(1, backbone_layers):
                # layer_list.append(nn.Linear(backbone_units, backbone_units))
                layer_list.append(nn.Conv2d(in_channels=backbone_units, out_channels=backbone_units, kernel_size=3, padding=1))
                layer_list.append(backbone_activation())
                if backbone_dropout > 0.0:
                    layer_list.append(torch.nn.Dropout(backbone_dropout))
            self.backbone = nn.Sequential(*layer_list)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # cat_shape = int(
        #     self.hidden_size + input_size if backbone_layers == 0 else backbone_units
        # )

        if backbone_layers == 0:
          cat_shape = torch.cat([torch.rand(input_size), torch.rand(self.hidden_size)], dim=1)
          cat_shape = cat_shape.shape
        else:
          cat_shape = [backbone_units,backbone_units]


        # self.ff1 = nn.Linear(cat_shape, hidden_size)
        self.ff1 = nn.Conv2d(in_channels=cat_shape[1], out_channels=hidden_size[1], kernel_size=3, padding=1)
        if self.mode == "pure":
            self.w_tau = torch.nn.Parameter(
                data=torch.zeros(1, self.hidden_size), requires_grad=True
            )
            self.A = torch.nn.Parameter(
                data=torch.ones(1, self.hidden_size), requires_grad=True
            )
        else:
            self.ff2 = nn.Conv2d(in_channels=cat_shape[1], out_channels=hidden_size[1], kernel_size=3, padding=1)
            self.time_a = nn.Conv2d(in_channels=cat_shape[1], out_channels=hidden_size[1], kernel_size=3, padding=1)
            self.time_b = nn.Conv2d(in_channels=cat_shape[1], out_channels=hidden_size[1], kernel_size=3, padding=1)

            # self.ff2 = nn.Linear(cat_shape, hidden_size)
            # self.time_a = nn.Linear(cat_shape, hidden_size)
            # self.time_b = nn.Linear(cat_shape, hidden_size)
        self.init_weights()

    def init_weights(self):
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                torch.nn.init.xavier_uniform_(w)

    def forward(self, input, hx, ts):
        x = torch.cat([input, hx], dim=1)
        if self.backbone_layers > 0:
            x = self.backbone(x)
        if self.sparsity_mask is not None:
            ff1 = F.linear(x, self.ff1.weight * self.sparsity_mask, self.ff1.bias)
        else:
            ff1 = self.ff1(x)


        # Cfc
        if self.sparsity_mask is not None:
            ff2 = F.linear(x, self.ff2.weight * self.sparsity_mask, self.ff2.bias)
        else:
            ff2 = self.ff2(x)
        ff1 = self.tanh(ff1)
        ff2 = self.tanh(ff2)
        t_a = self.time_a(x)
        t_b = self.time_b(x)


        t_interp = self.sigmoid(t_a * ts + t_b)
        if self.mode == "no_gate":
            new_hidden = ff1 + t_interp * ff2
        else:
            # print('flag1')
            # print('ff1: ',ff1.shape)
            # print('ff2: ',ff2.shape)
            # print('t_interp: ',t_interp.shape)
            new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2


        return new_hidden


