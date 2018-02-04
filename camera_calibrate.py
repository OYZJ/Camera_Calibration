#author: Ouyang Zhangjian
#date:2016/12/28

from __future__ import print_function

import numpy as np
import cv2

from Tkinter import *
from tkFileDialog import askdirectory

def selectPath(): #select the photos' directory
    path_ = askdirectory()
    path.set(path_)
def helpwindows():
    root = Tk()
    Message(root, text = "what is the camera caliberation?\n      Camera Caliberation In the process of image measurement and application of machine vision, in order to determine the relationship between the 3D geometric position of a certain point at spatial object surface and the corresponding point in the image, we must establish the geometric model of camera imaging, these geometric parameters of the model is the parameters of the camera .Under most conditions, these parameters must be obtained through experiment and calculation. This process is called camera calibration.").grid()
    root.mainloop()

root = Tk()
path = StringVar()
root.title("Camera Calliberation")
m = Menu(root) #menu part
root.config(menu = m)
filemenu = Menu(m)
m.add_cascade(label="file",menu=filemenu)
filemenu.add_command(label="open...",command = selectPath)
filemenu.add_separator()
filemenu.add_command(label = "exit",command = quit)
helpmenu = Menu(m)
m.add_cascade(label = "help",menu = helpmenu)
helpmenu.add_command(label="about...",command = helpwindows)

# local modules
from common import splitfn

# built-in modules
import os

if __name__ == '__main__':
    import sys
    import getopt
    from glob import glob

    args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug=', 'square_size='])
    args = dict(args)
    args.setdefault('--debug', './output/')
    args.setdefault('--square_size', 1.0)
    if not img_mask:
        img_mask = '../data/cc*.jpg'  # default
    else:
        img_mask = img_mask[0]

    img_names = glob(img_mask)
    debug_dir = args.get('--debug')
    if not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)
    square_size = float(args.get('--square_size'))

    pattern_size = (9, 9)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = 0, 0
    img_names_undistort = []
    for fn in img_names:
        print('processing %s... ' % fn, end='')
        img = cv2.imread(fn, 0)
        if img is None:
            print("Failed to load", fn)
            continue

        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if debug_dir:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            path, name, ext = splitfn(fn)
            outfile = debug_dir + name + '_chess.png'
            cv2.imwrite(outfile, vis)
            if found:
                img_names_undistort.append(outfile)

        if not found:
            print('chessboard not found')
            continue

        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

        print('ok')

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    # undistort the image with the calibration
    print('')
    for img_found in img_names_undistort:
        img = cv2.imread(img_found)

        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        outfile = img_found + '_undistorted.png'
        print('Undistorted image written to: %s' % outfile)
        cv2.imwrite(outfile, dst)

    cv2.destroyAllWindows()

Label(root,text = "import file's directory").grid(row = 1, column = 1)
Entry(root, textvariable = path).grid(row = 1, column = 0, sticky=N+S+W+E)
Label(root,text="This program is used to camera caliberation and calculate the nonlinear parameters.", fg="blue").grid(row=0, column=0,columnspan=2)
Label(root,text = "RMS", fg = "red").grid(row = 2, column = 0,sticky = N+W)
Label(root, text = rms).grid(row = 3, column = 0, sticky = N+W)
Label(root,text = "camera metrix", fg = "red").grid(row = 2, column = 1,sticky = N+W)
Label(root, text = camera_matrix).grid(row = 3, column = 1, sticky = N+W)
Label(root, text = "distortion coefficients", fg="red").grid(row = 4,column=0, sticky = N+W)
Label(root, text = dist_coefs.ravel()).grid(row = 5, column = 0, sticky =N+W)

root.mainloop()
