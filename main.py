import os
import sys
from random import randrange
from ale_py import ALEInterface
from ale_py.roms import Tetris

def main(rom_name, record_dir):
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
    while not ale.game_over():
        a = legal_actions[randrange(num_actions)]
        reward = ale.act(a)
        total_reward += reward

    print(f'Episode ended with score: {total_reward}')


if  __name__=="__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} record_dir")
        sys.exit()

    record_dir = sys.argv[1]
    main(Tetris, record_dir)