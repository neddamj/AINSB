/*
 * Connections: Arduino 3.3V -> Motor 3.3V
 *              Arduino GND -> BJT Emitter
 *              Arduino Pin 8 -> BJT Base
 *              BJT Collector -> Motor GND
*/

#define MOTOR 8

void setup() {
  pinMode(MOTOR, OUTPUT);
}

void loop() {
  // Turn motor on and off continuously
  digitalWrite(MOTOR, HIGH);
  delay(1000);
  digitalWrite(MOTOR, LOW);
  delay(1000);
}
