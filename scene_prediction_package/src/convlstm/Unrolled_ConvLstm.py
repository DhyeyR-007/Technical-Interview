# import general libraries
import torch
import torch.nn as nn

# import within repository dependencies
from Single_ConvLstm import Single_ConvLstm

# instantiating device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Unrolled_ConvLstm(nn.Module):

    '''
    Params:
    =================================================
    input_channels     = no. of channels in the input = int
    output_channels    = no. of channels desired in the output = int
    kernel_size        = size of filter that we will will convolve input image with = (int,int)
    padding            = amount of buffere space needed for convolving complete image without missing corners = (int,int)
    activation         = activation function selected (relu/tanh/sigmoid)
    size_image         = (height,width) = (int,int)


    
    Forward:
    ==================================================
    i/p:
    -----------
    X  = 5D input with shape (batch, channels, seq, height, width)
       = (no. of batches, no. of channels, no. of frames that form a sequence, frame height, frame width)

    o/p:
    -----------
    cell_output = 4D array with shape (batch, channels, seq, height, width)

    '''

    def __init__(self, input_channels, output_channels, kernel_size, padding, activation, size_image):
        super(Unrolled_ConvLstm, self).__init__()

        # initializing convlstm cell before unwinding it over time steps
        self.cell = Single_ConvLstm(in_channels=input_channels,out_channels=output_channels, kernel_size=kernel_size, 
        padding=padding, activation=activation, image_size=size_image)

        # initializing output channels desired
        self.output_channels = output_channels


    def forward(self, X):

        # Fetching dimensions from input (5D array)
        batch, channels, seq, height, width = X.size()

        # Hidden state and Cell state initialization (4D arrays)
        hidden_state, cell_state = [torch.zeros(batch, self.output_channels, height, width, device=device) for _ in range(2)]

        # Output initialization (5D array)
        cell_output = torch.zeros(batch, self.output_channels, seq, height, width, device=device)

        # Here convlstm cell is unrolled over a particular number( = sequence length) of time steps
        for t in range(seq):
            hidden_state, cell_state = self.cell(X[:,:,t], hidden_state, cell_state)
            cell_output[:,:,t] = hidden_state # saving the predictions in a sequence

        return cell_output

        