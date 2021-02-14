'''
    Author: Jordan Madden
    Usage: python rpi_i2c.py
'''

from smbus import SMBus

# Set the bus address and indicate I2C-1
addr = 0x08
bus = SMBus(1)

numb = 1

print("Enter 1 for ON or 0 for OFF")
while numb == 1:
    ledState = input(">>> ")
    
    if ledState == "1":
        bus.write_byte(addr, 0x01)
    elif ledState == "0":
        bus.write_byte(addr, 0x0)
    else:
        numb = 0