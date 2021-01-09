'''python tts.py'''
from gtts import gTTS 
import os

#Define the relevant constants and commands
LANGUAGE = "en"
FORWARD = "Go forward"
LEFT = "Turn left"
RIGHT = "Turn right"
STOP = "Stop moving"
PATH = "commands"

#If the folder is not present, generate the folder
if not os.path.exists(PATH):
    print("[INFO] Creating commands folder...")
    os.mkdir(PATH)
else:
    print("[INFO] Commands folder is already present...")

#Generate the commands and save them to the commands folder
print("[INFO] Saving the forward command audio files...\n")
speech = gTTS(text=FORWARD, lang=LANGUAGE, slow=False)
speech.save(os.path.join(PATH, "forward_command.mp3"))

print("[INFO] Saving the left command audio files...\n")
speech = gTTS(text=LEFT, lang=LANGUAGE, slow=False)
speech.save(os.path.join(PATH, "left_command.mp3"))

print("[INFO] Saving the right command audio files...\n")
speech = gTTS(text=RIGHT, lang=LANGUAGE, slow=False)
speech.save(os.path.join(PATH, "right_command.mp3"))

print("[INFO] Saving the stop command audio files...\n")
speech = gTTS(text=STOP, lang=LANGUAGE, slow=False)
speech.save(os.path.join(PATH, "stop_command.mp3"))


