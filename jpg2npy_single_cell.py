# MyYOLO Training dataset Generator
# Convert image data to xdata for model.fit()
import os
import cv2 as cv
import csv
import numpy as np
from icecream import ic
import glob
import random
import cfg
import iou_metric
import uuid

CONF_CH, XOFF_CH, YOFF_CH, W_CH, H_CH = 0,1,2,3,4

def gen_prim_ydata(inp_dir, random_flip=True, random_order=True, COLOR_AUG=True, BRIGHTNESS_AUG=True):
    files = []
    files.extend(glob.glob( os.path.join(inp_dir , '*.jpg') ))

    if random_order:
        random.shuffle(files)
    
    NIM = len(files)
    MODEL_INPUT_SIZE = cfg.MODEL_INPUT_SIZE
    x_data = np.zeros((NIM, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3), np.float32)
    Y_DIM = 15                                             # has_object,x,y,w,h, has_2nd_object,x2,y2,w2,h2, has_3rd_object, x3,y3,w3,h4
    y_data = np.zeros((NIM, 1, 1, Y_DIM), np.float32)
    nobj = 0

    for n,img_fn in enumerate(files):
        AUGM_FLIP = bool(random_flip) and random.choice([True,False])
        img = cv.imread(img_fn)
        if AUGM_FLIP:
            img = cv.flip(img, 1)       # Horizontal flip augmentation
        img = img.astype(np.float32)
        if BRIGHTNESS_AUG:
            contrast = random.uniform(-30., 30.)
            cfactor = 259.*(contrast+255.) / (255.*(259-contrast))
            img = cfactor * (img-128.) + 128.                             # random contrast
            img = img * random.uniform(0.8, 1.2)                         # random brightness
            img[img < 0.] = 0.
            img[img > 255.] = 255.
        img = img.astype(np.float32)
        if COLOR_AUG:
            # Satuation augmentation
            hsv = cv.cvtColor( img, cv.COLOR_BGR2HSV )
            value = random.uniform(0.77, 1.3)
            hsv[...,0] = hsv[...,0] * value
            value = random.uniform(0.77, 1.3)
            hsv[...,1] = hsv[...,1] * value
            hsv[hsv>255.] = 255.
            hsv[hsv<0.] = 0.
            img = cv.cvtColor( hsv, cv.COLOR_HSV2BGR )
        img = img.astype(np.float32)
        #img = cv.resize(img, (MODEL_INPUT_SIZE[1],MODEL_INPUT_SIZE[0]), interpolation=cv.INTER_NEAREST )
        img = cv.resize(img, (MODEL_INPUT_SIZE[1],MODEL_INPUT_SIZE[0]), interpolation=cv.INTER_AREA )

        x_data[n,...] = img
        y_data_1 = np.zeros( (1,1,Y_DIM), np.float32)
        y_data_1[:,:,XOFF_CH] = 0.
        y_data_1[:,:,YOFF_CH] = 0.
        label = os.path.splitext(img_fn)[0] + ".txt"
        if os.path.exists(label):
            with open(label, newline='\n') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                obj_idx = 0
                for row in spamreader:
                    _,x_str,y_str,w_str,h_str = row
                    box_x = float(x_str)
                    box_y = float(y_str)
                    assert box_x < 1. and box_x > 0.
                    assert box_y < 1. and box_y > 0.
                    if AUGM_FLIP:
                        box_x = 1. - box_x                                      # Horizontal flip augmentation
                    xy = np.array([box_x, box_y])

                    wh = np.array([float(w_str), float(h_str)])
                    #wh = np.array([1., 1.])

                    y_data_1[0, 0, CONF_CH] = 1.
                    xyoff = xy - [.5, .5]
                    if obj_idx == 0:
                        y_data_1[0, 0, 1:3] = xyoff
                        y_data_1[0, 0, 3:5] = wh*2
                    elif obj_idx == 1:
                        y_data_1[0, 0, 5] = 1.       # Indicate the 2nd object is valid
                        y_data_1[0, 0, 6:8] = xyoff
                        y_data_1[0, 0, 8:10] = wh*2            # range 0 to 1
                    elif obj_idx == 2:
                        y_data_1[0, 0, 10] = 1.       # Indicate the 3rd object is valid
                        y_data_1[0, 0, 11:13] = xyoff
                        y_data_1[0, 0, 13:15] = wh*2            # range 0 to 1

                    obj_idx += 1
                    nobj += 1
        y_data[n,...] = y_data_1
    np.save('ydata', y_data)
    x_data = x_data - 127.          # zero mean for resnet input
    np.save('xdata', x_data)
    #ic(x_data.shape)
    return files


def gen_prim_ydata_infinite(_):

    inp_dir = cfg.INPUT_DIR 
    random_flip   =1
    random_order  =1
    COLOR_AUG     =1
    BRIGHTNESS_AUG=1

    files = []
    files.extend(glob.glob( os.path.join(inp_dir , '*.jpg') ))

    if random_order:
        random.shuffle(files)
    
    NIM = len(files)
    MODEL_INPUT_SIZE = cfg.MODEL_INPUT_SIZE
    x_data = np.zeros((NIM, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3), np.float32)
    #Y_DIM = 15                                             # has_object,x,y,w,h, has_2nd_object,x2,y2,w2,h2, has_3rd_object, x3,y3,w3,h4
    Y_DIM = 1

    y_data = np.zeros((NIM, 1, 1, Y_DIM), np.float32)
    nobj = 0

    for n,img_fn in enumerate(files):
        AUGM_FLIP = bool(random_flip) and random.choice([True,False])
        img = cv.imread(img_fn)
        if AUGM_FLIP:
            img = cv.flip(img, 1)       # Horizontal flip augmentation
        img = img.astype(np.float32)
        if BRIGHTNESS_AUG:
            contrast = random.uniform(-30., 30.)
            cfactor = 259.*(contrast+255.) / (255.*(259-contrast))
            img = cfactor * (img-128.) + 128.                             # random contrast
            img = img * random.uniform(0.8, 1.2)                         # random brightness
            img[img < 0.] = 0.
            img[img > 255.] = 255.
        img = img.astype(np.float32)
        if COLOR_AUG:
            # Satuation augmentation
            hsv = cv.cvtColor( img, cv.COLOR_BGR2HSV )
            value = random.uniform(0.77, 1.3)
            hsv[...,0] = hsv[...,0] * value
            value = random.uniform(0.77, 1.3)
            hsv[...,1] = hsv[...,1] * value
            hsv[hsv>255.] = 255.
            hsv[hsv<0.] = 0.
            img = cv.cvtColor( hsv, cv.COLOR_HSV2BGR )
        img = img.astype(np.float32)
        img = cv.resize(img, (MODEL_INPUT_SIZE[1],MODEL_INPUT_SIZE[0]), interpolation=cv.INTER_AREA )

        #s1 = random.choice([-2,-1,0,1,2])
        #img = np.roll(img, s1, axis=1)
        #s0 = random.choice([-2,-1,0,1,2])
        #img = np.roll(img, s0, axis=0)

        #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        x_data[n,:,:,:] = img
        y_data_1 = np.zeros( (1,1,Y_DIM), np.float32)
        #y_data_1[:,:,XOFF_CH] = 0.
        #y_data_1[:,:,YOFF_CH] = 0.
        label = os.path.splitext(img_fn)[0] + ".txt"
        obj_idx = 0
        if os.path.exists(label):
            with open(label, newline='\n') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in spamreader:
                    _,x_str,y_str,w_str,h_str = row
                    box_x = float(x_str) #+ s1/float(MODEL_INPUT_SIZE[1])
                    box_y = float(y_str) #+ s0/float(MODEL_INPUT_SIZE[0])
                    assert box_x < 1. and box_x > 0.
                    assert box_y < 1. and box_y > 0.
                    if AUGM_FLIP:
                        box_x = 1. - box_x                                      # Horizontal flip augmentation
                    xy = np.array([box_x, box_y])

                    wh = np.array([float(w_str), float(h_str)])
                    #wh = np.array([.5, .5])

                    #y_data_1[0, 0, CONF_CH] = 1.
                    #xyoff = xy - [.5, .5]
                    #if obj_idx == 0:
                    #    y_data_1[0, 0, 1:3] = xyoff
                    #    y_data_1[0, 0, 3:5] = wh*2
                    #elif obj_idx == 1:
                    #    y_data_1[0, 0, 5] = 1.       # Indicate the 2nd object is valid
                    #    y_data_1[0, 0, 6:8] = xyoff
                    #    y_data_1[0, 0, 8:10] = wh*2            # range 0 to 1
                    #elif obj_idx == 2:
                    #    y_data_1[0, 0, 10] = 1.       # Indicate the 3rd object is valid
                    #    y_data_1[0, 0, 11:13] = xyoff
                    #    y_data_1[0, 0, 13:15] = wh*2            # range 0 to 1

                    obj_idx += 1
                    nobj += 1
        
        if obj_idx == 1:
            y_data_1[0, 0, 0] = 1.
        if obj_idx == 2:
            y_data_1[0, 0, 1] = 1.
        if obj_idx == 3:
            y_data_1[0, 0, 2] = 1.
        y_data[n,...] = y_data_1

    rand_name = str(uuid.uuid4())
    np.save('ydata_'+rand_name, y_data)
    x_data = x_data - 128.          # zero mean for resnet input
    x_data = x_data + np.random.uniform(low=-5.0, high=5.0, size=x_data.shape)
    np.save('xdata_'+rand_name, x_data)
    return files, 'xdata_'+rand_name, 'ydata_'+rand_name



def gen_sec_ydata():
    MODEL_DIM = float(cfg.MODEL_INPUT_SIZE[0])
    ydata = np.load('ydata.npy')    # (571,1,1,10)
    zdata = np.load('zdata.npy')    # (571,1,1,5)
    ydata2 = np.zeros( (zdata.shape[0],1,1,15) )
    for idx in range(zdata.shape[0]):                 # for each image
        if zdata[idx,0,0,0] > 0.5:
            xy = zdata[idx,0,0,1:3]
            wh = zdata[idx,0,0,3:5]

            gt_xy_a = ydata[idx,0,0,1:3]
            gt_wh_a = ydata[idx,0,0,3:5]

            gt_xy_b = ydata[idx,0,0,6:8]
            gt_wh_b = ydata[idx,0,0,8:10]

            gt_xy_c = ydata[idx,0,0,11:13]
            gt_wh_c = ydata[idx,0,0,13:15]

            ious_a = iou_metric.compute_iou(xy, wh, gt_xy_a, gt_wh_a)
            if ious_a > 0.5:
                ydata[idx,0,0,0] = 0.

            ious_b = iou_metric.compute_iou(xy, wh, gt_xy_b, gt_wh_b)
            if ious_b > 0.5:
                ydata[idx,0,0,5] = 0.

            ious_c = iou_metric.compute_iou(xy, wh, gt_xy_c, gt_wh_c)
            if ious_c > 0.5:
                ydata[idx,0,0,10] = 0.

    ydata2 = ydata
    np.save('ydata2', ydata2)

    # ┌─────────────────────────────────────────┐
    # │0│2│3│4│5│6│ 7  8  9 10 11 12 13 14  │
    # ├─┼─┼─┼─┼─┼─┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
    # │c│x│y│w│h│c│ x│ y│ w│ h  c  x  y  w  h │
    # └─┴─┴─┴─┴─┴─┴──┴──┴──┴──┴──┴──┴──┴───┴───┘

        

import multiprocessing as mp
import cv2

if __name__=='__main__':

    #with mp.Pool(10) as p:
    #    p.map(gen_prim_ydata_infinite, range(1))
    gen_prim_ydata_infinite(None)
    quit()

    #gen_sec_ydata()
    #quit()

    gen_prim_ydata(cfg.INPUT_DIR, random_flip=False, random_order=False, COLOR_AUG=False, BRIGHTNESS_AUG=False)
    xdata = np.load('xdata.npy')
    ydata = np.load('ydata.npy')
    print(xdata.shape)
    print(ydata.shape)
    quit()

    #yyy()
    #xdata1 = np.load('xdata1.npy')
    #print(xdata1.shape)
    #quit()

