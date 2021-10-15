import os
import sys
from random import randrange
from ale_py import ALEInterface
from ale_py.roms import Tetris

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


"""
<<<<<
TODO:
>>>>>

1. Write DeConv component of World Model Net

"""

def main(rom_name, record_dir):

    ### Torch Trolling
    #net = Net()
    #print(net)

    #params = list(net.parameters())
    #print(len(params))
    #print(params[0].size())  # conv1's .weight

    #input = torch.randn(1, 1, 32, 32)
    #out = net(input)
    #print(out)

    #net.zero_grad()
    #out.backward(torch.randn(1, 10))

    #print(out)


    ### ALE Stuff
    ale = ALEInterface()
    ale.setInt('random_seed', 123)

    # Enable screen display and sound output
    ale.setBool('display_screen', True)
    ale.setBool('sound', True)

    # Specify the recording directory and the audio file path
    ale.setString("record_screen_dir", record_dir) # Set the record directory
    ale.setString("record_sound_filename",
                    os.path.join(record_dir, "sound.wav"))

    ale.loadROM(rom_name)

    # Get the list of legal actions
    legal_actions = ale.getLegalActionSet()
    print(f'Legal Actions: {legal_actions}')
    num_actions = len(legal_actions)
    
    total_reward = 0
    arr = []
    itter = 0
    stack_tensor = None
    while not ale.game_over():

        stack_tensor = stackedFrames(ale, itter, stack_tensor)
        itter += 1

        a = legal_actions[randrange(num_actions)]
        arr.append(ale.getScreen())
        reward = ale.act(a)
        total_reward += reward
    print(arr)


    print(f'Episode ended with score: {total_reward}')


# Forms the (4, 3, 210, 160) dimensional tensor which serves as an input to our CNN
def stackedFrames(ale, itter, last_frames):

    frame_tensor = torch.Tensor(preProcessFrame(ale.getScreenRGB())).permute([2, 0, 1])
    frame_tensor = torch.unsqueeze(frame_tensor, 0)

    if itter == 0:
        stack_tensor = frame_tensor
    elif itter < 4:
        stack_tensor = torch.cat((last_frames, frame_tensor), dim=0)
    else:
        stack_tensor = torch.cat((last_frames[1:], frame_tensor), dim=0)

    print(f'Stack Tensor shape: {stack_tensor.shape}')

    return stack_tensor
 
def preProcessFrame(frame):
    """
    Prepares a Tensor for us 
    
    Args:
        frame (np.array): numpy array of 8-bit integers (0-255) that each 
                          represent a pixel colour in the Atari 2600. With
                          shape (210, 160).

    Returns:
        torch.Tensor: 
        
    """
    (n, d, channels) = frame.shape

    ret = torch.Tensor(n//2, d//2, channels)

    for c in range(channels):
        for i in range(n//2):
            for j in range(d//2):
                ret[i, j, c] = (frame[i, j, c] + frame[i, j+1, c] + frame[i+1, j, c] + frame[i+1, j+1, c]) // 4

    return ret


if  __name__=="__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} record_dir")
        sys.exit()

    record_dir = sys.argv[1]
    main(Tetris, record_dir)