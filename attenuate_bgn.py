from icecream import ic
import cv2 as cv
import glob
import numpy as np
import os
import csv
import uuid
import random
import shutil

X_CH = 0
Y_CH = 1
W_CH = 2
H_CH = 3
A_CH = 4
EN_CH = 5

# To label what makes emotion in a scene, which is not possible to automate, it's worth the man time.

def attenuate_bgn(img_path, center, outp_dir, wd=40):
    img = cv.imread(img_path)
    img = img.astype(np.float32)
    bgn = 128 + (cv.GaussianBlur(img,(3,3),0)/20.)-8.
    mask = np.zeros((img.shape[0], img.shape[1]))
    ctr_x = int(center[0])
    ctr_y = int(center[1])
    #wd = 40
    wd_int = int(round(wd/2))
    ht_top = 15
    ht_bottom = 90
    cv.rectangle(mask, (ctr_x-wd_int, ctr_y-ht_top), (ctr_x+wd_int, ctr_y+ht_bottom), (1.), -1)
    mask = cv.GaussianBlur(mask,(5,5),0)
    mask[mask>=0.5] = 1.
    mask[mask<0.5] = 0.
    mask = cv.GaussianBlur(mask,(5,5),0)
    mask = np.expand_dims(mask, axis=2)
    mask_inv = 1. - mask
    img = img * mask + bgn * mask_inv

    bname = os.path.basename(img_path)
    name = os.path.join(outp_dir, bname)
    cv.imwrite(name, img, [cv.IMWRITE_JPEG_QUALITY, 98])
    return img

def get_center(label):
    with open(label, newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            cls,x_str,y_str,w_str,h_str = row
            if int(cls)==0:
                box_x = float(x_str) * 256
                box_y = float(y_str) * 256
                return np.array((box_x, box_y))


def get_width(label):
    with open(label, newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            cls,x_str,y_str,w_str,h_str = row
            if int(cls)==1:
                ic(int(cls))
                cls,x_str,y_str,w_str,h_str = row
                ic(int(cls))
                box_w = float(w_str) * 256
                return box_w


#def gen_xydata(_):
#    inp_dir = R"C:\Users\faraday\my_yolo\flip_flop2\patch single\*.jpg"
#    inp_dir2 = R"C:\Users\faraday\my_yolo\flip_flop2\patch special\*.jpg"
#
#    outp_dir = R"C:\Users\faraday\my_yolo\flip_flop2\patch focus"
#
#    files = glob.glob(inp_dir)
#    files.extend(glob.glob(inp_dir2))
#    M = 2
#    N = len(files)*M
#    xdata = np.zeros((N,192,192,3))
#    ydata = np.zeros((N,1,1,2))
#    for fidx, fn in enumerate(files):    
#        center_lb = get_center(fn[0:-3]+'txt')
#        width_lb = get_width(fn[0:-3]+'txt')
#        for g in range(M):
#            offset = np.random.randint(low=-30, high=30, size=(2,))
#            center_img = center_lb + offset
#            img = attenuate_bgn(fn, center_img, outp_dir)
#            id = fidx*M+g
#            xdata[id,...] = cv.resize(img,(192,192),interpolation=cv.INTER_AREA)
#
#            offset[0] = np.minimum(offset[0], 20)
#            offset[0] = np.maximum(offset[0], -20)
#
#            offset[1] = np.minimum(offset[1], 20)
#            offset[1] = np.maximum(offset[1], -20)
#
#            ydata[id,:,:,0] = offset[0]/256.
#            ydata[id,:,:,1] = offset[1]/256.
#
#    rand_name = str(uuid.uuid4())
#    rand_name = rand_name[0:8]
#    np.save('ydata_'+rand_name, ydata)
#    xdata = xdata - 128.          # zero mean for resnet input
#    xdata = xdata.astype(np.float32)
#    np.save('xdata_'+rand_name, xdata)
#    ic(xdata.shape)
    


def render(fn, x, y, wd, ht, ang, AUGM=True, FLIP=False):
    img = cv.imread(fn)
    img = img.astype(np.float32)
    BRIGHTNESS_AUG=AUGM
    if BRIGHTNESS_AUG:
            contrast = random.uniform(-30., 30.)
            cfactor = 259.*(contrast+255.) / (255.*(259-contrast))
            img = cfactor * (img-128.) + 128.                             # random contrast
            img = img * random.uniform(0.8, 1.2)                         # random brightness
            img[img < 0.] = 0.
            img[img > 255.] = 255.
    if FLIP:
        img = cv.flip(img, 1)

    mask = np.zeros((img.shape[0], img.shape[1]))
    cv.rectangle(mask, (128-int(wd), 128-int(ht)), (128+int(wd), 128+int(ht)), (1.), -1)        
    mask = rotate_image(mask, ang)
    mask = translate_image(mask, (x,y))
    mask = cv.GaussianBlur(mask,(27,27),0)
    mask[mask>=0.5] = 1.
    mask[mask<0.5] = 0.
    mask = cv.GaussianBlur(mask,(13,13),0)
    mask = np.expand_dims(mask, axis=2)
    img_mk = mask * img + (1.-mask) * 128.
    return img_mk


def gen_xydata(_):
    inp_dir = R"C:\Users\faraday\my_yolo\flip_flop2\rotation set\*.jpg"
    inp_dir2 = R"C:\Users\faraday\my_yolo\flip_flop2\rotation set 2\*.jpg"
    inp_dir3 = R"C:\Users\faraday\my_yolo\flip_flop2\rotation set 3\*.jpg"
    files = glob.glob(inp_dir)
    files.extend(glob.glob(inp_dir2))
    files.extend(glob.glob(inp_dir3))
    random.shuffle(files)
    n_obj = 0
    for fn in files:
        xywha = np.load(fn[0:-3]+'npy')
        for g in range(xywha.shape[0]):
            if xywha[g,5] > .5:
                n_obj += 1

    xdata = np.zeros((n_obj,192,192,3), dtype=np.float32)
    ydata = np.zeros((n_obj,1,1,6))
    obj_idx = 0
    for fidx, fn in enumerate(files):
        xywha = np.load(fn[0:-3]+'npy')
        for g in range(xywha.shape[0]):
            do_flip = random.choice([True, False])
            if xywha[g,5] > .5:
                rand_x = float(np.random.uniform( -20, 20, [1,] ))
                rand_y = float(np.random.uniform( -20, 20, [1,] ))
                rand_wd = float(np.random.uniform( -10, 13, [1,] ))
                rand_ht = float(np.random.uniform( -20, 25, [1,] ))
                rand_ang = float(np.random.uniform( -15, 15, [1,] ))      # swing toward vertical
                if do_flip == False:
                    x = xywha[g,0] + rand_x
                else:
                    x = -xywha[g,0] + rand_x 
                y = xywha[g,1] + rand_y
                wd = xywha[g,2] + rand_wd
                ht = xywha[g,3] + rand_ht
                if do_flip == False:
                    ang = xywha[g,4] + rand_ang
                else:
                    ang = -xywha[g,4] + rand_ang
                img = render(fn, x, y, wd, ht, ang, AUGM=True, FLIP=do_flip)
                xdata[obj_idx,...] = cv.resize(img, (192,192), interpolation=cv.INTER_AREA)

                ang_off = np.minimum(rand_ang, 5)
                ang_off = np.maximum(rand_ang, -5)

                rand_x = np.minimum(rand_x, 14)
                rand_x = np.maximum(rand_x, -14)

                rand_y = np.minimum(rand_y, 20)
                rand_y = np.maximum(rand_y, -20)

                ydata[obj_idx,:,:,0] = rand_x/40.
                ydata[obj_idx,:,:,1] = rand_y/40.

                ydata[obj_idx,:,:,2] = xywha[g,2]/100.
                ydata[obj_idx,:,:,3] = xywha[g,3]/250.

                ydata[obj_idx,:,:,5] = (xywha[g,3]/xywha[g,2])/10.

                ydata[obj_idx,:,:,4] = ang_off/90.
                obj_idx += 1

    xd,yd = manual_tag()
    xdata = np.concatenate((xdata, xd), axis=0)
    ydata = np.concatenate((ydata, yd), axis=0)

    rand_name = str(uuid.uuid4())
    rand_name = rand_name[0:8]
    np.save(R'E:\tmp\ydata_'+rand_name, ydata)
    xdata = xdata - 128.          # zero mean for resnet input
    np.save(R'E:\tmp\xdata_'+rand_name, xdata)





def gen_data(fn, offset):
    outp_dir = R"C:\Users\faraday\my_yolo\flip_flop2\patch focus"
    xdata = np.zeros((1,192,192,3))
    img = attenuate_bgn(fn, offset, outp_dir)
    xdata[0,...] = cv.resize(img, (192,192), interpolation=cv.INTER_AREA)
    xdata = xdata - 128.
    return xdata, img

def show_att():
    img_path=R"C:\Users\faraday\my_yolo\flip_flop2\patch single\TW_1400_17.jpg"
    fn = img_path
    outp_dir=R"C:\Users\faraday\my_yolo\flip_flop2\New folder"
    center_lb = (-30.,0.) + get_center(fn[0:-3]+'txt')
    attenuate_bgn(img_path, center_lb, outp_dir, wd=40)


def rotate_image(image, angle):    
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, float(angle), 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_AREA)
    return result

def translate_image(image, dist):
    mat = np.float32([
	    [1, 0, dist[0]],
	    [0, 1, -dist[1]]
    ])
    #mat = np.array([[1, 0, x], [0, 1, -y]], dtype=np.float32)
    result = cv.warpAffine(image, mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result




def manual_tag():
    inp_dir3 = R"C:\Users\faraday\my_yolo\flip_flop2\rotation set 3\*.jpg"
    files = glob.glob(inp_dir3)
    random.shuffle(files)
    nimg = len(files)
    xdata = np.zeros((nimg,192,192,3), dtype=np.float32)
    ydata = np.zeros((nimg,1,1,6))
    for idx, fn in enumerate(files):    
        xywha = np.load(fn[0:-3]+'npy')
        img = render(fn, 0., 0., 126., 126., 0., AUGM=True, FLIP=False)
        xdata[idx,...] = cv.resize(img, (192,192), interpolation=cv.INTER_AREA)
        xystep = 0. - xywha[0, 0:2]
        #whstep = 126. - xywha[0, 2:4]
        xystep = np.minimum(xystep, [12,12])
        xystep = np.maximum(xystep, [-12,-12])
        #whstep = np.minimum(whstep, [10,10])
        #whstep = np.maximum(whstep, [-10,-10])
        ydata[idx,0,0,0:2] = xystep/40
        ydata[idx,0,0,W_CH] = xywha[0,W_CH]/100      #  whstep/100.
        ydata[idx,0,0,H_CH] = xywha[0,H_CH]/250
        ydata[idx,0,0,A_CH] = 0./90.
        ydata[idx,0,0,5] = (xywha[0,H_CH]/xywha[0,W_CH])/10.
    return xdata, ydata


if __name__=='__main__':

    #gen_xydata(None)
    #quit()

    screen = np.zeros((768,768,3), dtype=np.float32)

    files = glob.glob(R"C:\Users\faraday\my_yolo\flip_flop2\patch dual\*.jpg")
    fidx = 0
    lb_dir = R"C:\Users\faraday\my_yolo\flip_flop2\rotation set 2"

    while True:
        print('fidx:', fidx)
        fn = files[fidx]
        img = cv.imread(fn)
        img = img.astype(np.float32)
        xywha = np.zeros( (3,6), dtype=np.float32)
        xywha[0,0:6] = 0,0,40,60,-14,0
        xywha[1,0:6] = -45, 55, 40, 60, -14, 0
        idx = -1
        lb_path=os.path.join(lb_dir,os.path.basename(fn))
        lb_path=lb_path[0:-4]+'.npy'
        if os.path.exists(lb_path):
            xywha=np.load(lb_path)
    
        while True:
            img_disp = img
            for k in range(3):
                if xywha[k,EN_CH] > .5:
                    pos = xywha[k,:]
                    dist = pos[0:2]
                    lyr2 = np.zeros((img.shape[0], img.shape[1]))
                    cv.rectangle(lyr2, (128-int(pos[2]), 128-int(pos[3])), (128+int(pos[2]), 128+int(pos[3])), (1.), 2)
                    lyr2 = rotate_image(lyr2, pos[4])
                    lyr2 = translate_image(lyr2, dist)
                    lyr2 = np.expand_dims(lyr2, axis=2)
                    lyr2 = lyr2/2
                    img_disp = (1.-lyr2)*img_disp + lyr2*(0.,140.,0.)
            if idx >= 0:
                if xywha[idx,EN_CH] > .5:
                    pos = xywha[idx,:]
                    dist = pos[0:2]
                    lyr2 = np.zeros((img.shape[0], img.shape[1]))
                    cv.rectangle(lyr2, (128-int(pos[2]), 128-int(pos[3])), (128+int(pos[2]), 128+int(pos[3])), (1.), 2)
                    lyr2 = rotate_image(lyr2, pos[4])
                    lyr2 = translate_image(lyr2, dist)
                    lyr2 = np.expand_dims(lyr2, axis=2)
                    img_disp = (1.-lyr2)*img_disp + lyr2*(0.,255.,0.)

            img_disp = img_disp/255
            bname = os.path.basename(fn)
            screen[:]=0
            screen[128:640,128:640,:] = cv.resize(img_disp,None,fx=2,fy=2)

            font = cv.FONT_HERSHEY_SIMPLEX #cv.FONT_HERSHEY_PLAIN
            
            fontScale = 1
            color = (255, 255, 0)
            thickness = 1
            span = 30
            org = (10, 30)
            cv.putText(screen, bname, org, font, fontScale, color, thickness, cv.LINE_AA)
            org = (400, 30)
            cv.putText(screen, '%04d/%d'%(fidx+1, len(files)), org, font, fontScale, color, thickness, cv.LINE_AA)
            org = (5, 450+span*0)
            cv.putText(screen, 'X:%+04d'%xywha[idx,X_CH], org, font, fontScale, color, thickness, cv.LINE_AA)
            org = (5, 450+span*1)
            cv.putText(screen, 'Y:%+04d'%xywha[idx,Y_CH], org, font, fontScale, color, thickness, cv.LINE_AA)
            org = (5, 450+span*2)
            cv.putText(screen, 'W:%+04d'%xywha[idx,W_CH], org, font, fontScale, color, thickness, cv.LINE_AA)
            org = (5, 450+span*3)
            cv.putText(screen, 'H:%+04d'%xywha[idx,H_CH], org, font, fontScale, color, thickness, cv.LINE_AA)
            org = (5, 450+span*4)
            cv.putText(screen, 'A:%+4.1f'%xywha[idx,A_CH], org, font, fontScale, color, thickness, cv.LINE_AA)
            
            cv.imshow('', screen)

            k = cv.waitKeyEx(0)
            
            if k==27:
                quit()
            if k==3014656:
                print('del')
                xywha[idx, EN_CH]=0
                for q in range(3):
                    idx+=1
                    if idx>2:
                        idx=0
                    if xywha[idx, EN_CH]>.5:
                        break
            if k == ord('d'):
                fidx+=1
                if fidx >= len(files):
                    fidx = len(files)-1
                break
            if k == ord('a'):
                fidx-=1
                if fidx<0:
                    fidx=0
                break
            if k == ord('w'):
                for n1 in range(3):
                    if xywha[n1, EN_CH] < 0.5:
                        print('New object')
                        xywha[n1, EN_CH] = 1.
                        xywha[n1, X_CH]=0
                        xywha[n1, Y_CH]=0
                        xywha[n1, W_CH]=40
                        xywha[n1, H_CH]=80
                        xywha[n1, A_CH]=0
                        idx=n1
                        break
            if xywha[idx, EN_CH] > 0.5:
                if idx >= 0:
                    if k==2490368:            # arrow up
                        xywha[idx,Y_CH] += 2
                    if k==2621440:            # arrow down
                        xywha[idx,Y_CH] -= 2
                    if k==2424832:   #ord('a'):
                        xywha[idx,X_CH] -= 2
                    if k==2555904:
                        xywha[idx,X_CH] += 2

                if k==2162688:   # page down
                    xywha[idx,A_CH] += 2
                    print(xywha[idx,A_CH])
                if k==2228224:    # page up
                    xywha[idx,A_CH] -= 2
                    print(xywha[idx,A_CH])

                if k==91 or k==ord('3') or k==2293760:
                    xywha[idx,H_CH] -= 1

                if k==93 or k==ord('1') or k==2359296:
                    xywha[idx,H_CH] += 1


                if k==ord('=') or k==ord('e'):
                    xywha[idx,W_CH] += 1

                if k==ord('-') or k==ord('q'):
                    xywha[idx,W_CH] -= 1

    
                if k == ord(' '):
                    bname=os.path.basename(fn)
                    path=os.path.join(lb_dir,bname)
                    np.save(path[0:-4], xywha)
                    shutil.copy(fn, lb_dir)
                    print('Saved')
                    idx=-1
            if k==9:
                for q in range(3):
                    idx+=1
                    if idx>2:
                        idx=0
                    if xywha[idx, EN_CH]>.5:
                        break
                #bname='out'
                #cv.imwrite(path, img_mk, [cv.IMWRITE_JPEG_QUALITY, 98])
