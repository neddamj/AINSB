'''
    Author: Jordan Madden
    Usage: python tactile_feedback/rpi_i2c.py
'''

from smbus import SMBus

# Set the bus address and indicate I2C-1
addr = 0x08
bus = SMBus(1)

print("Enter a number from 0 to 3")
while True:
    ledState = input(">>> ")
    
    if ledState == "1":
        bus.write_byte(addr, 0x01)
    elif ledState == "0":
        bus.write_byte(addr, 0x0)
    elif ledState == "2":
        bus.write_byte(addr, 0x2)
    elif ledState == "3":
        bus.write_byte(addr, 0x3)
    else:
        break