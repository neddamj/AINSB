/*
 *  Author: Jordan Madden
 *  Title: Tactile Feedback
 *  Date: 10/1/2021
 *  Description: The atmega328p must receive data from the raspberry pi via the UART serial
 *               communication protocol. Based on the received data it muct then activate 
 *               corresponding vibration motor, thus alerting the user as to which direction
 *               they should move in or whether they should move at all.
 */

#define F A1
#define L A2
#define R A3 

void setup() {
  // Set the baud rate of the microcontroller
  Serial.begin(9600);

  // Set the mode of the I/O pins
  pinMode(F, OUTPUT);
  pinMode(L, OUTPUT);
  pinMode(R, OUTPUT);
}

void loop() {
  // Recieve message from Raspberry Pi
  recMessage = Serial.read();
  
  if(recMessage == 1){
    forward(); 
  }
  else if(recMessage == 2){
    left();
  }
  else if(recMessage == 3){
    right();
  }

  delay(1000);
}

void forward(){
  digitalWrite(F, HIGH);
}

void left(){
  digitalWrite(L, HIGH);
}

void right(){
  digitalWrite(R, HIGH);
}

void stopMoving(){
  digitalWrite(F, HIGH);
  digitalWrite(L, HIGH);
  digitalWrite(R, HIGH);
}
