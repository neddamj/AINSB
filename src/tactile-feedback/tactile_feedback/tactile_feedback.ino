/*
 *  Author: Jordan Madden
 *  Title: Tactile Feedback
 *  Date: 10/1/2021
 *  Description: The atmega328p must receive data from the raspberry pi via the UART serial
 *               communication protocol. Based on the received data it muct then activate 
 *               corresponding vibration motor, thus alerting the user as to which direction
 *               they should move in or whether they should move at all.
 */

#define forward
#define left
#define right
#define stopMoving

void setup() {
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:

}
