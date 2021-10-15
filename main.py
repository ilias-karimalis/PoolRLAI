import os
import sys
from random import randrange
from ale_py import ALEInterface
from ale_py.roms import Tetris

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    frame_tensor = torch.Tensor(ale.getScreenRGB()).permute([2, 0, 1])
    frame_tensor = torch.unsqueeze(frame_tensor, 0)

    if itter == 0:
        stack_tensor = frame_tensor
    elif itter < 4:
        frame_tensor = torch.unsqueeze(torch.Tensor(ale.getScreenRGB()).permute([2, 0, 1]), dim=0)
        stack_tensor = torch.cat((last_frames, frame_tensor), dim=0)
    else:
        stack_tensor = torch.cat((last_frames[1:], frame_tensor), dim=0)

    print(f'Stack Tensor shape: {stack_tensor.shape}')
    return stack_tensor


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.pixelEmbeding = nn.Linear()

        self.conv1 = nn.Conv2d(in_channels=64,out_channels=128, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if  __name__=="__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} record_dir")
        sys.exit()

    record_dir = sys.argv[1]
    main(Tetris, record_dir)