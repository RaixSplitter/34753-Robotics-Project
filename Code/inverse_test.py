from sympy import symbols, cos, sin, Matrix, atan2, acos, sqrt, pi, Array
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

theta_1, theta_2, theta_3, theta_4= symbols('theta_1 theta_2 theta_3 theta_4')
dh_conv = {
    1 : {'theta': theta_1, 'd': 50/1000, 'a': 0, 'alpha': 90, 'type': 'revolute'},
    2 : {'theta': theta_2, 'd': 0, 'a': 93/1000, 'alpha': 0, 'type': 'revolute'},
    3 : {'theta': theta_3, 'd': 0, 'a': 93/1000, 'alpha': 0, 'type': 'revolute'},
    4 : {'theta': theta_4, 'd': 0, 'a': 50/1000, 'alpha': 0, 'type': 'revolute'},
    5 : {'x' : -15/1000, 'y' : 45/1000, 'z' : 0, 'type': 'transformation'}
}

def inv_kinematics(x,y,z,phi, dh_conv):
    """
    Hardcoded inverse kinematics for the robot arm
    """
    
    theta1 = atan2(y, x)
    
    a2 = dh_conv[2]['a']
    a3 = dh_conv[3]['a']
    a4 = dh_conv[4]['a']
    d1 = dh_conv[1]['d']
    r = sqrt(x**2 + y**2)
    r2 = cos(phi)*a4
    #Given r = r1 + r2
    r1 = r - r2
    z1 = d1
    z3 = sin(phi)*a4
    #Given z = z1 + z2 + z3
    z2 = z - z1 - z3
    c23 = sqrt(z2**2 + r1**2)
    
    D = (c23**2 - a2**2 - a3**2)/(-2*a2*a3)
    theta3 = -(atan2(sqrt(1-D**2), D) - pi)
    theta2 = atan2(z2, r1) - atan2(a3*sin(theta3), a2 + a3*cos(theta3))
    theta4 = phi - theta2 - theta3
    
    return float(theta1), float(theta2), float(theta3), float(theta4)

def rad_to_ang(rad):
    return [int(r/pi*2 * 308 + 512) for r in rad]


move_joints([512, 512, 512, 512])
while True:
    reach_error = False
    
    try:    values = map(float, input("input:").split(" "))
    except: print("Value Conversion Error"); continue
    
    try:                   rad = inv_kinematics(*values, dh_conv)
    except Exception as e: print(e); continue
    
    for i,r in enumerate(rad):
        if r == None:
            print("Error: joint %s is nan!" % (i))
            reach_error = True
    if reach_error: continue
        
    ang = rad_to_ang(rad)
    for i,a in enumerate(ang):
        if a < 150 or a > 874:
            print("Error: joint %s is out of allowed range [%s]!" % (i,a))
            reach_error = True
    if reach_error: continue
    
    move_joints(ang)
    

# 0 0 0.1 0

# min = 150
# -90 = -pi/2 = 204 
#   0 =   0   = 512 (+308)
#  90 =  pi/2 = 820
# max = 874

# capture1 = cv2.VideoCapture(0)