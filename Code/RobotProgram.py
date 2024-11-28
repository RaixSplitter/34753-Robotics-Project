import dynamixel_sdk as dxl
import cv2
import csv
import torch
from torch import nn
import msvcrt
import numpy as np
from random import choice
import matplotlib.pyplot as plt
from skimage.filters import median, gaussian, threshold_otsu
from skimage.morphology import erosion, dilation, disk

# robot settings
joint_speed                   = 40
DEVICENAME                    = 'COM3'
camera_pos                    = [512, 390, 370, 480]
# [512, 330, 120, 772] # close
# [512, 390, 370, 480] # camera
# [512, 188, 497, 552] # far

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
DXL_IDS                       = [1,2,3,4]
BAUDRATE                      = 1_000_000
TORQUE_ENABLE                 = 1
TORQUE_DISABLE                = 0
portHandler = dxl.PortHandler(DEVICENAME)
packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)
portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)

for DXL_ID in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_MARGIN, 0)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_MARGIN, 0)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_SLOPE, 32)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_SLOPE, 32)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_MOVING_SPEED, joint_speed)

# misc
def getch():
    return msvcrt.getch().decode()

# robot functions
def boot_robot(robot_on : bool):
    if robot_on:
        portHandler.openPort()
        portHandler.setBaudRate(BAUDRATE)
    else:
        portHandler.closePort()

def move_joints(goals):
    is_success = True
    for i,goal in enumerate(goals):
        dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_IDS[i], ADDR_MX_GOAL_POSITION, goal)
        if dxl_comm_result != dxl.COMM_SUCCESS: 
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
    # check for errors
    # if not is_success:
    #     boot_robot(False)
    #     raise Exception("Error")
    
    # wait for reaching goal
    at_joint = 0
    while True:
        # Read present position
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_IDS[at_joint], ADDR_MX_PRESENT_POSITION)
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))

        # print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (DXL_IDS[at_joint], goals[at_joint], dxl_present_position))

        if abs(goals[at_joint] - dxl_present_position) <= 10: # threshold = 10
            at_joint += 1
        if at_joint == len(goals):
            break
    
# cell segmentation
def get_cells(img_gray):
    # dilation_disk = disk(2)
    dilation_disk = np.ones((5,5))
    # Detect edges
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

    # Extract line coordinates
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        # Separate horizontal and vertical lines based on the angle (theta)
        if 0 <= theta < np.pi / 4 or np.pi * 3 / 4 <= theta <= np.pi:
            vertical_lines.append(rho)
        else:
            horizontal_lines.append(rho)

    # Add image borders as lines
    height, width = img_gray.shape
    horizontal_lines.extend([0, height])  # Top and bottom borders
    vertical_lines.extend([0, width])    # Left and right borders

    # Sort lines to determine grid layout
    horizontal_lines = sorted(horizontal_lines)
    vertical_lines = sorted(vertical_lines)

    # Crop spaces between lines
    cropped_regions = []

    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            # Get the coordinates of the bounding box
            y1 = int(horizontal_lines[i])
            y2 = int(horizontal_lines[i + 1])
            x1 = int(vertical_lines[j])
            x2 = int(vertical_lines[j + 1])

            # Crop the region
            cropped = img_gray[y1:y2, x1:x2]

            # Filter regions by size (to exclude empty or grid-only areas)
            if cropped.shape[0] > 50 and cropped.shape[1] > 50:  # Adjust size threshold as needed
                cropped_regions.append(dilation((cv2.resize(cropped, dsize=(28,28), interpolation=cv2.INTER_CUBIC) < 100) * 1.0, dilation_disk))
    return np.array(cropped_regions)

# define region network
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Your code here!
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.batchN1 = nn.BatchNorm2d(16)
        self.batchN2 = nn.BatchNorm2d(32)
        self.drop    = nn.Dropout(p=0.5)
        
        self.pool  = nn.MaxPool2d(2, 2)
        self.flat = nn.Flatten()
        
        self.fc1   = nn.Linear(800, 256)
        self.fc2   = nn.Linear(256, 128)
        self.fc3   = nn.Linear(128, 3)
        self.log   = nn.LogSoftmax(dim=1)
        self.Lrelu = nn.LeakyReLU()
        

    def forward(self, x):
        # Your code here!
        x = x.view(-1,1,28,28)
        x = self.drop(self.batchN1(self.pool(self.Lrelu(self.conv1(x)))))
        x = self.drop(self.batchN2(self.pool(self.Lrelu(self.conv2(x)))))
        x = self.flat(x)
        x = self.Lrelu(self.fc1(x))
        x = self.Lrelu(self.fc2(x))
        x = self.fc3(x)
        x = self.log(x)
        return x

# tic-tac-toe
class v_agent():
    def __init__(self, has_x):
        self.v = self.get_v_table()
        # if is_first: self.evaluate = max
        # else:        self.evaluate = min
        if has_x: 
            self.xo = "X"
            self.evaluate = max
        else:
            self.xo = "O"
            self.evaluate = min

    def get_v_table(self):
        with open('state_values.csv', 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            return {row["State"]:float(row["Value"]) for row in csvreader}
        
    def get_valid_actions(self, state):
        return [i for i,xo in enumerate(state) if xo == " "]
    
    def get_new_state(self, state, action):
        return state[:action] + self.xo + state[action+1:]
    
    def get_action_values(self, state):
        valid_actions = self.get_valid_actions(state)
        return {action : self.v[self.get_new_state(state, action)] for action in valid_actions}
        
    def choose_action(self, state : str):
        action_values = self.get_action_values(state)
        target = self.evaluate(action_values.values())
        next_action = choice([action for action, v in action_values.items() if v == target])
        return next_action
# program
def run_program():
    # initialisze
    boot_robot(True)
    move_joints(camera_pos) # reset to start position
    robot_cam = cv2.VideoCapture(0)
    model = Model()
    model.load_state_dict(torch.load('model.pth', weights_only=True))
    model.eval()
    agent = v_agent(has_x=True)
    state_conversion = {0:" ", 1:"O", 2:"X"}
    ver_pos = {0: "Top", 1:"Middle", 2:"Bottom"}
    hor_pos = {0: "Left", 1:"Middle", 2:"Right"}
    
    # run program
    while True:
        # move to camera position
        move_joints(camera_pos)
        while True:
            _result, img = robot_cam.read() # Read an image
            cv2.imshow("ImageWindow", img) # Display the image
            # if (cv2.waitKey(2) >= 0): # If the user presses a key, exit while loop
            #     break
            k = cv2.waitKey(33)
            if k==27:    # Esc key to stop
                return
            elif k==-1:  # normally -1 returned,so don't print it
                continue
            else:
                break
        # print("Press any key to take photo! (or press ESC to quit!)")
        # if getch() == chr(0x1b): # if escape
        #     break
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # get regions
        cropped_regions = get_cells(img_gray)
        cropped_tensors = torch.from_numpy(cropped_regions).float()
        print(cropped_tensors.shape)
        if cropped_tensors.shape[0] != 9:
            print("Error in segmentation")
            continue
        fig,ax = plt.subplots(3,3)
        fig.suptitle(f"Cells ")
        axes = []
        for x in ax:
            for y in x:
                axes.append(y)
        for ax,region in zip(axes,cropped_regions):
            ax.imshow(region,cmap="gray")
        plt.axis("off")
        plt.show()
        
        # convert regions to state
        raw_output = model(cropped_tensors)
        # print(raw_output)
        raw_state = raw_output.max(1)[1].detach().tolist()
        state = "".join([state_conversion[i] for i in raw_state])
        print(state)
        if abs(state.count("X") - state.count("O")) > 1:
            print("Invalid bord state:")
            continue

        # evaluate position
        action = agent.choose_action(state)
        print(f"{ver_pos[action//3]} {hor_pos[action%3]}")
        print(action) 
        
    # cv2.destroyAllWindows() # Close window
    cv2.VideoCapture(0).release() # Release video device
    boot_robot(False)

run_program()

