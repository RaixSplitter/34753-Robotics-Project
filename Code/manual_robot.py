import dynamixel_sdk as dxl
import cv2

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

for DXL_ID in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_MARGIN, 0)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_MARGIN, 0)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_SLOPE, 32)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_SLOPE, 32)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_MOVING_SPEED, 40)

def move_joint(i):
    # dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_IDS[i], ADDR_MX_GOAL_POSITION, get_goal(i))
    dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_IDS[i], ADDR_MX_GOAL_POSITION, dxl_goal_position[i])
    if dxl_comm_result != dxl.COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))

### Restrained
# joint_check       = [0, 1, 1, 1]
# get_goal = lambda x: (dxl_goal_position[x]*616//100+204) * joint_check[x] + (dxl_goal_position[x]*1023//100)*(1 - joint_check[x])
# dxl_goal_position = [50, 50, 50, 50]
# ang = 50
# for i,goal_ang in enumerate(dxl_goal_position):
#     ang += (goal_ang - 50) * joint_check[i]
#     if ang < 0 or ang > 100:
#         raise Exception(f"Error: goal {i+1} breaks the robot")
### Unrestrained
dxl_goal_position = [512, 415, 315, 505]
for i in range(len(DXL_IDS)):
    move_joint(i)

# Open video device
capture1 = cv2.VideoCapture(0)
joint = 0
while True:
    ret, img = capture1.read() # Read an image
    cv2.imshow("ImageWindow", img) # Display the image
    # if (cv2.waitKey(2) >= 0): # If the user presses a key, exit while loop
    #     break
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break
    elif k==-1:  # normally -1 returned,so don't print it
        continue
    elif k==97: # a
        # dxl_goal_position[joint-1] = min(dxl_goal_position[joint-1] + 1,100)
        dxl_goal_position[joint-1] = min(dxl_goal_position[joint-1] + 1,1023)
        print(joint, dxl_goal_position[joint-1])
        move_joint(joint-1)
    elif k==100: # d
        # dxl_goal_position[joint-1] = max(dxl_goal_position[joint-1] - 1,0)
        dxl_goal_position[joint-1] = max(dxl_goal_position[joint-1] - 1,0)
        print(joint, dxl_goal_position[joint-1])
        move_joint(joint-1)
    elif k in [49,50,51,52]:
        joint = k - 48
        print(joint)
    else:
        # print(k)
        print(dxl_goal_position)

cv2.destroyAllWindows() # Close window
cv2.VideoCapture(0).release() # Release video device
for DXL_ID in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)

portHandler.closePort()

