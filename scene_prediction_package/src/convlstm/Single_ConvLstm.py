# ConvLSTM cell architecture adapted and inspired from

# @article{shi2015convolutional,
#   title={Convolutional LSTM network: A machine learning approach for precipitation nowcasting},
#   author={Shi, Xingjian and Chen, Zhourong and Wang, Hao and Yeung, Dit-Yan and Wong, Wai-Kin and Woo, Wang-chun},
#   journal={Advances in neural information processing systems},
#   volume={28},
#   year={2015}
# }


# import general libraries
import torch
import torch.nn as nn


class Single_ConvLstm(nn.Module):

    '''
    Params:
    =================================================
    in_channels     = no. of channels in the input = int
    out_channels    = no. of channels desired in the output = int
    kernel_size     = size of filter that we will will convolve input image with = (int,int)
    padding         = amount of buffere space needed for convolving complete image without missing corners = (int,int)
    activation      = activation function selected (relu/tanh/sigmoid)
    image_size      = (height,width) = (int,int)


    
    Forward:
    ==================================================
    i/p:
    -----------
    X          = input to the LSTM cell
    H_previous = Hidden state from previous cell
    C_previous = Cell state from the previous cell

    o/p:
    -----------
    H = updated hidden state
    C = updated cell state

    '''


    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, image_size):
        super(Single_ConvLstm, self).__init__()  

        # Select activation
        if activation == "relu" or activation == None:
            self.activation = torch.relu 
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sigmoid":
            self.activation = torch.sigmoid     
    
        # Reference: https://github.com/ndrplz/ConvLSTM_pytorch
        self.layer1 = nn.Conv2d(in_channels=in_channels + out_channels, out_channels=4 * out_channels, kernel_size=kernel_size, padding=padding)    

        # Weight initilaization for Hadamard Products in LSTM gating mechanisms (according to shi2015convolutional)
        self.Wt_input_gate = nn.Parameter(torch.Tensor(out_channels, *image_size))
        self.Wt_forget_gate = nn.Parameter(torch.Tensor(out_channels, *image_size))
        self.Wt_output_gate = nn.Parameter(torch.Tensor(out_channels, *image_size))


    def forward(self, X, H_previous, C_previous):

        # Reference: https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.layer1(torch.cat([X, H_previous], dim=1))
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        # Input gate formulation
        i_gate = torch.sigmoid(i_conv + self.Wt_input_gate * C_previous)

        # Forget gate formulation
        f_gate = torch.sigmoid(f_conv + self.Wt_forget_gate * C_previous)

        # Cell state
        cell_state = f_gate * C_previous + i_gate * self.activation(C_conv)

        # Output gate formulation
        o_gate = torch.sigmoid(o_conv + self.Wt_output_gate * cell_state)

        # Hidden State
        hidden_state = o_gate * self.activation(cell_state)

        return hidden_state, cell_state
    


