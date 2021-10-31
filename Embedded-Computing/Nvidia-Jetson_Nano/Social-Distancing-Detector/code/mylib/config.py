# base path to YOLO directory
MODEL_PATH = "yolo"
# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

#===============================================================================
#=================================\CONFIG./=====================================
""" Below are your desired config. options to set for real-time inference """
# To count the total number of people (True/False).
People_Counter = True
# Set the threshold value for total violations limit.
Threshold = 15
# Set url = 0 for webcam.
url = '0'
# Set if GPU should be used for computations; Otherwise uses the CPU by default.
USE_GPU = True
# Define the max/min safe distance limits (in pixels) between 2 people.
MAX_DISTANCE = 80
MIN_DISTANCE = 50
#===============================================================================
#===============================================================================
