import time
import os
import logging
logging.getLogger("imported_module").setLevel(logging.ERROR)

# Dictionary to store all audio commands to be  played
command = {}
command["Forward"] = "omxplayer commands/forward_command.mp3"
command["Right"] = "omxplayer commands/right_command.mp3"
command["Left"] = "omxplayer commands/left_command.mp3"
command["Stop"] = "omxplayer commands/stop_command.mp3"

os.system(command["Forward"])
time.sleep(1)
os.system(command["Right"])
time.sleep(1)
os.system(command["Left"])
time.sleep(1)
os.system(command["Stop"])
time.sleep(1)