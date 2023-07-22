# import general libraries
import torch
import torch.nn as nn

# import within repository dependencies
from Unrolled_ConvLstm import Unrolled_ConvLstm


class Sequence_to_Sequence(nn.Module):

    '''
    Params:
    =================================================
    desired_channels = no. of channels in the input and needed in the output image = int
    kernels              = no. of hidden channels in the form of kernels to use = int
    kernel_size          = size of filter that we will will convolve input image with = (int,int)
    padding              = amount of buffer space needed for convolving complete image without missing corners = (int,int)
    activation           = activation function selected (relu/tanh/sigmoid)
    frame_size           = incoming image's (height,width) = (int,int)
    LAYERS               = desired number of unrolled convlstm layers i.e. number of enoder & decoder layers = int

                        (i) Ideal number should be 3 since First layer is to cater to incoming input channels; 
                            Second layer is encoder; and Third layer is decoder, the second and third layers are a 
                            proper encoder-decoder since they perform seq2seq operation in same number of channels. 
                            3 is ideal no. of layers also from a computational expense point of view.

                        (ii) The LAYERS is independent of the last conv2d layer since that is necessary to equate 
                            to the number of output channels
    

    
    Forward:
    ==================================================
    i/p:
    -----------
    X  = 5D input with shape (batch, channels, seq, height, width)
       = (no. of batches, no. of channels, no. of frames that form a sequence, frame height, frame width)

    o/p:
    -----------
    final_output = desired channel binary predicted image (desired_channels, height, width)

    '''

    def __init__(self, desired_channels, kernels, kernel_size, padding, activation, frame_size, LAYERS):

        super(Sequence_to_Sequence, self).__init__()

        # initializing seq2seq model sequential
        self.architecture = nn.Sequential()

        # Subsequent layer chunks (convlstm + Batch Norm (BN))
        for layer in range(1, LAYERS+1):
                if layer == 1:
                    ip_ch = desired_channels  # First layer chunk (convlstm + Batch Norm (BN)) (to cater to input channels)
                else: 
                    ip_ch = kernels

                self.architecture.add_module( f"Layer_{layer}", Unrolled_ConvLstm(
                                                input_channels=ip_ch, output_channels=kernels,
                                                padding=padding, kernel_size=kernel_size,
                                                size_image=frame_size, activation=activation))
                self.architecture.add_module(f"BN_{layer}", nn.BatchNorm3d(num_features=kernels)) 


        # Final conv layer 
        self.finalconv = nn.Conv2d(in_channels=kernels, out_channels=desired_channels, kernel_size=kernel_size, padding=padding)


    def forward(self, X): 

        # Propagating through seq2seq architecture
        arch_output = self.architecture(X)

        # Mapping the extracted features from seq2seq architecture to the desired output format
        # giving out only the last output frame 
        finalconv_output = self.finalconv(arch_output[:,:,-1])

        # Used to squash the output values of the final convolutional layer into the range of [0, 1] to apply BCE loss in backpropagation.
        final_output = torch.sigmoid(finalconv_output)
        
        return final_output  #nn.Sigmoid()(output)

    