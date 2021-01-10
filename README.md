# AINSB - AI Navigation System for the Blind
The aim of this project is to develop a prototype device that assists the blind/visually impaired to navigate safely in various environments. This project is done for partial fulfillment of the BSc. Electronics Engineering at UWI Mona.

## Project Overview
According to the project requirements, the primary objectives of the project are to:
* Identify various objects in the path of the user
* Estimate the distance of those objects from the user
* Provide auditory and/or tactile feedback to direct the user

The secondary/stretch objectives include:
* Apply face recognition to identify persons familiar to the user of the device 
* Accept voice commands from the user 

As such, the software implementation of the project up to this point has been divided into the following section:
* Depth Estimation
* Face Recognition
* Object Detection
* Text to Speech
* Tactile Feedback

Each of the previously mentioned sections contain the code and files that were used during the R&D process for each of those tasks. Those previous sections, and the approaches taken within each of the sections will be expanded upon.

## Object Detection
2 approaches were taken to complete this task. First, a YOLOv3 object detector was tested using the Deep Neural Network(dnn) module in OpenCV and then various models were tested through the Tensorflow 2 Object Detection API(OD API). From the OD API, the models that were settled upon were the SSD MobileNet V2 and the EFficientDet D0.

## Face Recognition
As before, 2 approaches were taken here. In the first approach, I developed a custom face recognition pipeline using the OpenCV dnn module and scikit-learn and in the second approach I used the face_recognition library that is available in python.

## Text to Speech
Here, I made use of Google's Text to Speech(TTS) API and the playsound library. The TTS API allowed me to convert text into .mp3 files and the playsound library allowed me to playback these .mp3 files as audio.

##Tactile Feedback
Here, the atmega328p should receive data from the raspberry pi using the UART protocol and it should trigger motors based on the data it recieves from the pi.
