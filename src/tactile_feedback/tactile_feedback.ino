/*
 *  Author: Jordan Madden
 *  Title: Tactile Feedback
 *  Date: 10/1/2021
 *  Description: The atmega328p must receive data from the raspberry pi via the I2C serial
 *               communication protocol. Based on the received data it muct then activate 
 *               corresponding vibration motor, thus alerting the user as to which direction
 *               they should move in or whether they should move at all.
 */

#include <Wire.h>

#define L A3
#define F A2
#define R A1

void setup() {
  // Set the baud rate for serial communication
  Serial.begin(9600);
  
  // Join the I2C bus as a slave device
  Wire.begin(0x8);

  // Call receive event when data is received
  Wire.onReceive(receiveEvent);

  // Set the mode of the I/O pins
  pinMode(F, OUTPUT);
  pinMode(L, OUTPUT);
  pinMode(R, OUTPUT);
}

void loop(){
}

void receiveEvent(int howMany){
   while(Wire.available()){
    char recMessage = Wire.read();

    commandUser(recMessage);
   }
}

void forward(){
  digitalWrite(F, HIGH);
  digitalWrite(L, LOW);
  digitalWrite(R, LOW);
  Serial.println("Go Forward");
}

void turnLeft(){
  digitalWrite(L, HIGH);
  digitalWrite(F, LOW);
  digitalWrite(R, LOW);
  Serial.println("Turn Left");
}

void turnRight(){
  digitalWrite(R, HIGH);
  digitalWrite(L, LOW);
  digitalWrite(F, LOW);
  Serial.println("Turn Right");
}

void stopMoving(){
  digitalWrite(F, HIGH);
  digitalWrite(L, HIGH);
  digitalWrite(R, HIGH);
  Serial.println("Stop Moving");
}

void hapticsOff(){
  digitalWrite(L, LOW);
  digitalWrite(F, LOW);
  digitalWrite(R, LOW);
  Serial.println("Motors Off");  
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
  else if(recMessage == 0){
    stopMoving();
  }
  else{
    hapticsOff();
  }
}
