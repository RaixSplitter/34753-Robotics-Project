import dynamixel_sdk as dxl
import cv2
from time import sleep

ADDR_MX_TORQUE_ENABLE         = 24
ADDR_MX_CW_COMPLIANCE_MARGIN  = 26
ADDR_MX_CCW_COMPLIANCE_MARGIN = 27
ADDR_MX_CW_COMPLIANCE_SLOPE   = 28
ADDR_MX_CCW_COMPLIANCE_SLOPE  = 29
ADDR_MX_GOAL_POSITION         = 30
ADDR_MX_MOVING_SPEED          = 32
ADDR_MX_PRESENT_POSITION      = 36
ADDR_MX_PUNCH                 = 48
PROTOCOL_VERSION              = 1.0
DXL_IDS                       = [1,152,3,4]
BAUDRATE                      = 1_000_000
TORQUE_ENABLE                 = 1
TORQUE_DISABLE                = 0
# auto connect
for i in range(10):
    try:
        portHandler = dxl.PortHandler("COM%s" % i)
        portHandler.openPort()
        print("Connected to [COM%s]" % i)
        break
    except:
        continue
else:
    raise Exception("No connection!")
packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)
portHandler.setBaudRate(BAUDRATE)

while True:
    sleep(1)
    print([packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_MX_PRESENT_POSITION)[0] for DXL_ID in DXL_IDS])

# Robot test:
# Cam : [512, 365, 370, 500]
pos_dict = {
    "bm"  : [512, 320, 300, 620],
    "c"   : [512, 275, 390, 585],
    "tm"  : [512, 210, 515, 560],
    
    "br"  : [485, 330, 290, 620],
    "mr"  : [487, 280, 385, 585],
    "tr"  : [490, 210, 515, 570],
    
    "bl"  : [539, 330, 290, 620],
    "ml"  : [537, 280, 385, 585],
    "tl"  : [534, 210, 515, 570],
}

for DXL_ID in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_MARGIN, 0)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_MARGIN, 0)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_SLOPE, 32)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_SLOPE, 32)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_MOVING_SPEED, 40)

def move_joints(goals):
    for i,goal in enumerate(goals):
        dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_IDS[i], ADDR_MX_GOAL_POSITION, goal)
        if dxl_comm_result != dxl.COMM_SUCCESS: 
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
    
    # wait for reaching goal
    at_joint = 0
    while True:
        # Read present position
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_IDS[at_joint], ADDR_MX_PRESENT_POSITION)
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))

        if abs(goals[at_joint] - dxl_present_position) <= 10: # threshold = 10
            at_joint += 1
        if at_joint == len(goals):
            break

capture1 = cv2.VideoCapture(0)
at_joint = 0
while True:
    move_joints([512, 365, 370, 500])
    ret, img = capture1.read() # Read an image
    cv2.imshow("ImageWindow", img) # Display the image
    
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break
    elif k==-1:  # normally -1 returned,so don't print it
        continue
    else:
        name,pos = list(pos_dict.items())[at_joint]
        print(name)
        move_joints(pos)
        at_joint = (at_joint + 1) % len(pos_dict)
        sleep(3)
