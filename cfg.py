MODEL_INPUT_SIZE = (192, 192)
INPUT_DIR = R'C:\Users\faraday\my_yolo\flip_flop2\patch single'
#INPUT_DIR = R"C:\Users\faraday\my_yolo\flip_flop2\New folder - Copy"
#INPUT_DIR = R"C:\Users\faraday\my_yolo\flip_flop2\test set"

LEARNING_RATE = 0.0261/400
BATCH_SIZE = 42

COLOR_AUG          = 1            # apply to training only
BRIGHTNESS_AUG     = 1
GEO_AUG            = 1

#PREDICT_DIR = R'C:\Users\faraday\my_yolo\flip_flop2\New folder'
#PREDICT_DIR = R"C:\Users\faraday\my_yolo\flip_flop2\patch"

OUTPUT_DIR = R'C:\Users\faraday\my_yolo\flip_flop2\result'

# 0  1  2  3  4   5  6  7  8  9   10 11 12 13 14
# c  x  y  w  h   c  x  y  w  h    c  x  y  w  h
