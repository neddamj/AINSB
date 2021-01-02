'''python audio_commands.py'''
import os
import time
from collections import defaultdict
from playsound import playsound 

def playback(motion_command, cmds):
    #Play audio recording of the given command
    playsound(cmds[motion_command])

if __name__ == "__main__":
    #Define relevant file paths
    COMMAND_PATH = 'commands'
    FWD = 'forward_command.mp3'
    LEFT = 'left_command.mp3'
    RIGHT = 'right_command.mp3'
    STOP = 'stop_command.mp3'

    #Paths to audio files
    forwardPath = os.path.join(COMMAND_PATH, FWD)
    leftPath = os.path.join(COMMAND_PATH, LEFT)
    rightPath = os.path.join(COMMAND_PATH, RIGHT)
    stopPath = os.path.join(COMMAND_PATH, STOP)

    #Store the commands in a dictionary
    commands = defaultdict()
    commands["forward"] = forwardPath
    commands["left"] = leftPath
    commands["right"] = rightPath
    commands["stop"] = stopPath

    #Play the commands
    playback("forward", commands)
    time.sleep(0.5)

    playback("left", commands)
    time.sleep(0.5)

    playback("right", commands)
    time.sleep(0.5)

    playback("stop", commands)
    time.sleep(0.5)
