/*
 *  Author: Jordan Madden
 *  Title: Tactile Feedback
 *  Date: 10/1/2021
 *  Description: The atmega328p must receive data from the raspberry pi via the UART serial
 *               communication protocol. Based on the received data it muct then activate 
 *               corresponding vibration motor, thus alerting the user as to which direction
 *               they should move in or whether they should move at all.
 */

#include <Arduino.h>
#include <Wire.h>

#define F A1
#define L A2
#define R A3 

void setup() {
  // Set the baud rate for serial communication
  Serial.begin(9600);
  
  // Join the I2C bus as a slave device
  Wire.begin(0x08);

  // Call receive event when data is received
  Wire.onReceive(receiveEvent);

  // Set the mode of the I/O pins
  pinMode(F, OUTPUT);
  pinMode(L, OUTPUT);
  pinMode(R, OUTPUT);
}

void loop() {
  Serial.println("-----------");
}

void receiveEvent(int howMany){
   while(Wire.available()){
    char recMessage = Wire.read();

    commandUser(recMessage);
   }
}

void forward(){
  digitalWrite(F, HIGH);
  Serial.println(" Go Forward");
}

void turnLeft(){
  digitalWrite(L, HIGH);
  Serial.println("Turn Left");
}

void turnRight(){
  digitalWrite(R, HIGH);
  Serial.println("Turn Right");
}

void stopMoving(){
  digitalWrite(F, HIGH);
  digitalWrite(L, HIGH);
  digitalWrite(R, HIGH);
  Serial.println("Stop Moving");
}

void commandUser(unsigned char recMessage){
  if(recMessage == 1){
    forward();
  }
  else if(recMessage == 2){
    turnLeft();
  }
  else if(recMessage == 3){
    turnRight();
  }
  else{
    stopMoving();
  }
}
