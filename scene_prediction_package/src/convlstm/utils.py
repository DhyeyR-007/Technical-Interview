# import libraries
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import io
import imageio
from ipywidgets import widgets, HBox, VBox
import os, shutil
from PIL import Image, ImageFilter
from skimage.measure import find_contours
from IPython.display import display, Image
from moviepy.editor import *
from natsort import natsorted
import glob



### open npy file and save images
def npy_to_images (npy_filename, rgb_image_dir):
    dat = np.load(npy_filename)
    print("numpy data shape: ", dat.shape)
    for i in range(1,len(dat),1):
        vol = dat[i, ..., 0]
        top_down_view = vol.max(axis=2)
        plt.imshow(top_down_view)
        plt.axis('off')
        plt.savefig(rgb_image_dir + str(i) + '.png', bbox_inches="tight",pad_inches=0)
        plt.close()


### images to video
def img_to_video (rgb_image_dir, video_address):
    img = cv2.imread(rgb_image_dir + '1.png') 
    height,width,layers = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec selection
    video = cv2.VideoWriter(video_address, fourcc, 1, (width, height))
    lisk = []
    for i in range(1,len(os.listdir(rgb_image_dir)),1):
      imgg = cv2.imread(rgb_image_dir + str(i) + '.png')
      video.write(imgg)
    cv2.destroyAllWindows()
    video.release()


## folder file remover
def empty_the_repo(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def frame_skipping(cap, out, skip_factor):
    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Write the frame to the output video
        if index % skip_factor == 0:
            out.write(frame)
        index += 1
    out.release()
    cap.release()
    cv2.destroyAllWindows()



def reverse_video(cap, out):
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    for frame in reversed(frames): # Write the frames in reverse order to the output video
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()




def random_flip(cap, out, flip_code):
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, flip_code)  # change this line to add diffeent variations of augementations
                                                # 0 --> flipping around x-axis
                                                # 1 or any +ve value --> flipping around y-axis
                                                #-1 or any -ve value --> flipping around both x and y axes
            out.write(frame) # Write the frame to the output video
        out.release()
        cap.release()
        cv2.destroyAllWindows()




def combiner(directory_path, prefix = "augment"):
    L = []
    files_starting_with_prefix = []
    for filename in os.listdir(directory_path):
        if filename.startswith(prefix):
            files_starting_with_prefix.append(os.path.join(directory_path, filename))
    for path in sorted(files_starting_with_prefix):
        video = VideoFileClip(path)
        L.append(video)
    final_clip = concatenate_videoclips(L)

    final_path_combined = directory_path + 'COMBINED.mp4'
    final_clip.to_videofile(final_path_combined, remove_temp=False)
    cap_combined = cv2.VideoCapture(directory_path + 'COMBINED.mp4')
    frames_count = cap_combined.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap_combined.get(cv2.CAP_PROP_FPS)
    print(f'{final_clip.duration/60} mins.; {final_clip.duration} secs.; {frames_count} frames; {fps} fps')

    return final_path_combined



def apply_augmentations_and_combine(input_video_path, video_repo, augmentations):
    for i, (augment, args) in enumerate(augmentations):
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = video_repo + f'augment_{i+1}.mp4'
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        augment(cap=cap, out=out, **args)
    combined_path= combiner(video_repo)

    return combined_path



## splitting video to images sequentially
def splitter (combined_video_path, dir_video_to_img):
    cap = cv2.VideoCapture(combined_video_path)
    ret, frame = cap.read()
    frame_num = 0
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)  # any fps value can be put here to get video frames to
                                     # speed up or slow down, defaults using original fps


    while ret:
        # Check if the current frame is a multiple of fps
        if frame_num % fps== 0:
            cv2.imwrite(os.path.join(dir_video_to_img,f'{count}.png'), frame)
            count += 1

        # Read the next frame
        ret, frame = cap.read()
        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()




## grayscale - thresholding
def sharpen_image(image):
    blurred = cv2.GaussianBlur(image, (0, 0),3)
    sharpened = cv2.addWeighted(image, 99.5, blurred, -0.5, 0)
    return sharpened


def get_bnw(img_add):
    originalImage = cv2.imread(img_add)
    denoise = cv2.fastNlMeansDenoisingColored(originalImage,None,10,10,7,20)
    grayImage1 = cv2.cvtColor(denoise, cv2.COLOR_RGB2GRAY)
    _, im_bw = cv2.threshold(grayImage1, 0, 455, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im_bw = sharpen_image(im_bw)
    return grayImage1, im_bw


def convert_to_grayscale (dir_video_to_img, dir_img_to_grayscale):
    for i in range(0, len(os.listdir(dir_video_to_img)),1):
        _,sharpened_image = get_bnw(os.path.join(dir_video_to_img,f'{i}.png'))
        # sharpened_image = cv2.resize(sharpened_image, img_size)
        plt.axis('off')
        plt.imshow(sharpened_image, cmap='gray', interpolation='bicubic')
        plt.savefig(os.path.join(dir_img_to_grayscale,f'gray_{i}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()


## dataset creator
def create_dataset (dir_img_to_grayscale, batches, length_of_sequence, img_size, shuffling=False):

        assert (batches * length_of_sequence) == len(os.listdir(dir_img_to_grayscale)), "batches * length_of_sequence ≠ length of provided directory"

        X_lis = []
        for ii in range(0,len(os.listdir(dir_img_to_grayscale)),1):
                image = cv2.imread(os.path.join(dir_img_to_grayscale,f'gray_{ii}.png'))
                resized_image = cv2.resize(image,img_size)
                single_channel = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
                expanded_single_channel = np.expand_dims(single_channel, axis=-1)
                X_lis.append(expanded_single_channel)
        input_data = np.array(X_lis)
        input_data = input_data.reshape(batches,length_of_sequence,input_data.shape[1],input_data.shape[2])
        if shuffling:
            np.random.shuffle(input_data)
        return input_data


## To preprocess and prepare the data for training, validation & testing batches as needed by the DataLoader.
def custom(inp):

    # Last 10 frames are target
    target = np.array(inp)[:,10:]   

    # Convert the input batch of data samples to a tensor
    batch_tensor = torch.tensor(inp)

    # add in a channel dimension
    batch_tensor = batch_tensor.unsqueeze(1)

    # Send the tensor to the appropriate device (GPU or CPU) and Normalize pixel values
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_tensor = (batch_tensor / 255.0).to(device)

    # Randomly pick a frame index between 1 and 9 (inclusive) for the target
    test_num = np.random.randint(1, 10)

    # Return the 10 frames preceding the random index as input and the random frame as target
    input_frames = batch_tensor[:, :, test_num:test_num+10]
    target_frame = batch_tensor[:, :, test_num+10]

    return input_frames, target_frame, batch_tensor, target


## Plot data
def plot(list_1, label_1 , list_2, label_2, x_label, y_label):

    # Create x-axis values
    if len(list_1) != len(list_2):
        assert "unequal size of plot lists"
    x = range(len(list_1))

    # Plot
    plt.plot(x, list_1, 'r', label= label_1)
    plt.plot(x, list_2, 'b', label= label_2)

    # Add labels and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    # Display the plot
    plt.show()


###################################################################################
## For gif and overlapped gif creation and systematic storage of the same

def overlap (img1,img2):
    # Load the grayscale images
    ground_truth = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_RGB2GRAY)

    predicted = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    predicted = cv2.cvtColor(predicted, cv2.COLOR_RGB2GRAY)

    # Create an empty 3-channel image
    overlay = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 3), dtype=np.uint8)

    # Assign the grayscale images to the red and blue channels respectively
    overlay[:,:,2] = ground_truth  # Red channel
    overlay[:,:,1] = predicted    # Yellow channel

    return overlay


def gif_to_frames (add):
    fname = add
    gif = imageio.mimread(fname)
    frames_target = [cv2.cvtColor(img1, cv2.COLOR_RGB2BGR) for img1 in gif]
    return frames_target


def gif_and_superimposed_gif_creator(FILE, target, output, original_dataset_type, testing_dataset_type, epochs, centerloss_flag):

      if centerloss_flag == True:
        specific_filepath = f'gifs_{epochs}_epochs_' + original_dataset_type + '_training_centerloss/'
      else:
        specific_filepath = f'gifs_{epochs}_epochs_' + original_dataset_type + '_training_without_centerloss/'


      FILE = os.path.join(FILE,specific_filepath)
      if not os.path.exists(FILE):
            os.mkdir(FILE)

      count = 0
      for tgt, out in zip(target, output):  # Loop over samples

          # Write target video as gif (LEFT)
          with io.BytesIO() as gif:
              imageio.mimsave(gif, tgt, "GIF", duration=200,loop = 0)
              target_gif = gif.getvalue()
              with open(FILE + testing_dataset_type + f'_target_batch_{count+1}.gif', 'wb') as file:
                  file.write(target_gif)

          # Write output video as gif (RIGHT)
          with io.BytesIO() as gif:
              imageio.mimsave(gif, out, "GIF", duration=200,loop = 0)
              output_gif = gif.getvalue()
              with open(FILE +  testing_dataset_type + f'_output_batch_{count+1}.gif', 'wb') as file:
                  file.write(output_gif)

          # gif labels
          target_label = widgets.Label(value=f"Ground truth for batch no. {count + 1}", layout=widgets.Layout(margin='0 0 0 50'))
          predicted_label = widgets.Label(value=f"Predicted for batch no. {count + 1}", layout=widgets.Layout(margin='0 0 0 50'))

          # Combine label and image vertically using VBox layout
          target_vbox = VBox([target_label, widgets.Image(value=target_gif)], layout=widgets.Layout(margin='0 90px'))
          predicted_vbox = VBox([predicted_label, widgets.Image(value=output_gif)], layout=widgets.Layout(margin='0 20px'))

          # Display the combined Combine target and predicted VBox containers using HBox container
          display(HBox([target_vbox, predicted_vbox]))

          count += 1


      tgt,out = None, None
      count = 0
      for op,tar in zip (sorted(glob.glob(FILE + testing_dataset_type  + "_output*")), sorted(glob.glob(FILE +  testing_dataset_type + "_target*"))):

          # ground truth (target) and predicted (output) gif to frames
          frames_output = gif_to_frames(op)
          frames_target = gif_to_frames(tar)

          # print('FO',len(frames_output))
          # print('FT',len(frames_target))

          # creating a list where we add all the overlayyed frames
          overlayyed = []
          for i in range(0,len(frames_target)):
              overlayyed.append(overlap(frames_target[i],frames_output[i]))

          # Create a list to store resized frames
          resized_frames = []

          # Resize and preserve color of each frame
          for frame in overlayyed:
              resized_frame = cv2.resize(frame, (1000, 1000))
              resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB color mode
              resized_frames.append(resized_frame)


          # Save the frames as GIF
          with imageio.get_writer(FILE +  testing_dataset_type + f"_batch_{count+1}.gif", mode="I", duration=200,loop = 0) as writer:
              for idx, frame in enumerate(resized_frames):
                  writer.append_data(frame)

          count += 1

      op,tar = None, None


###################################################################################




