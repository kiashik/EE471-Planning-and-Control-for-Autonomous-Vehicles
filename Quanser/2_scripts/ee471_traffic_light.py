## Traffic light example
# This example initializes a Traffic Light and runs through the different
# commands to control it.  
# Modify the light_ip variable to have the correct IP of the Traffic light.
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
# imports

from pal.products.traffic_light import TrafficLight
import time
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 

# Provide the Traffic Light IP address
## -- -- -TRAFFIC LIGHT IP- -- -- 
light_ip = '192.168.2.80'
# -- -- -- -- -- -- -- -- -- -- --

# Change to be normal Traffic Pattern
# Possibilities include 'Traffic', 'Quick', 'Red', 'Yellow', 'Green'
# If TRAFFIC, red for 30s, green for 30s, and yellow for 3s
# If QUICK, iterate through each light, 3 seconds each
# Colors denote solid states
pattern = 'Quick'

# Time to run (seconds)
run_time = 300

# Initialize a Traffic Light with its corresponding IP
light = TrafficLight(light_ip)

# Check the status of the lights. 0 - No lights, 1 - Red LED, 
# 2 - Yellow LED, 3 - Green LED
status = light.status()
print("Traffic Light Status is: " + status)

try:
    match(pattern):
        case('Traffic'):
            automaticMode = light.auto()
            print('Running Lights in Traffic Mode: RED -> 30s, GREEN -> 30s, YELLOW -> 3s')
        case('Quick'):
            customMode = light.timed(3, 3, 3)
            print('Running Lights in Quick Mode: RED -> 3s, GREEN -> 3s, YELLOW -> 3s')
        case('Red'):
            redMode = light.red()
            print('Red LED On Mode')
        case('Green'):
            greenMode = light.green()
            print('Green LED On Mode')
        case('Yellow'):
            yellowMode = light.yellow()
            print('Yellow LED On Mode')
        case('Radiator Springs'):
            start_time = time.time()
            while time.time() - start_time < run_time:
                rs = light.yellow()
                time.sleep(1.5)
                rs = light.off()
                time.sleep(1.5)
                rs = light.yellow()
                time.sleep(1.5)
                rs = light.off()
                time.sleep(1.5)
                rs = light.yellow()
                time.sleep(1.8)
                rs = light.off()
                time.sleep(1.5)
        case _:
            run_time = 0
            print('Invalid Light Pattern Entered')
    time.sleep(run_time)
    turnOff = light.off()
    print(turnOff)
except:
    turnOff = light.off()
    print(turnOff)