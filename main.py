import math
import random
import threading
import time
import tkinter
from tkinter import *
from tkinter import filedialog
from tkinter.messagebox import showinfo, askyesno
from tkinter.ttk import Progressbar

import cv2
import numpy as np
from PIL import Image, ImageTk
from rembg import remove
from ttkbootstrap import Style

import tiler
import conf


def Showimage(cv2_img, canva, layout="fit", TYPE=0):
    """
   Displaying OpenCV Images in the Canvas Widget of Tkinter
   !!!! This includes a global variable named "imgTK"!!!! Please do not use this variable name elsewhere!
   "canvas": Tkinter canvas variable used for display
   "layout": Display format. Options are:
   "fill": Image automatically adapts to canvas size and fills it completely, possibly causing distortion.
   "fit": Displays the image as large as possible without distorting it, leaving edges blank if necessary, depending on the canvas size.
   "type": Display type. By default, type 0 will be recorded in the snapshot, while other types will not be.
    """
    if TYPE == 0:
        fastFlash(cv2_img)

    global imgTK
    canvas_width = int(canva.winfo_reqwidth())
    canvas_height = int(canva.winfo_reqheight())
    sp = cv2_img.shape
    cv_height = sp[0]  # height(rows) of image
    cv_width = sp[1]  # width(columns) of image
    if layout == "fill":
        imgCV = cv2.resize(cv2_img, (canvas_width, canvas_height), interpolation=cv2.INTER_AREA)
    elif layout == "fit":
        if float(cv_width / cv_height) > float(canvas_width / canvas_height):
            imgCV = cv2.resize(cv2_img, (canvas_width, int(canvas_width * cv_height / cv_width)),
                               interpolation=cv2.INTER_AREA)
        else:
            imgCV = cv2.resize(cv2_img, (int(canvas_height * cv_width / cv_height), canvas_height),
                               interpolation=cv2.INTER_AREA)
    else:
        imgCV = cv2_img
    imgCV2 = cv2.cvtColor(imgCV, cv2.COLOR_BGR2RGBA)  # Convert colors from BGR to RGBA
    current_image = Image.fromarray(imgCV2)  # Converting images to Image objects
    imgTK = ImageTk.PhotoImage(image=current_image)  # Converting image objects to imageTK objects
    imgX = (700 - imgTK.width()) / 2
    imgY = (500 - imgTK.height()) / 2
    canva.create_image(imgX, imgY, anchor=NW, image=imgTK)


def findIMG(event=None):
    #  Open the local file browser and select the image to display
    global canvas
    filename.set(filedialog.askopenfilename())  # Get the selected file
    global IMG
    global IMGs
    global position
    IMGs.clear()
    position = 0

    IMG = cv2.imread(filename.get())
    Showimage(IMG, canvas, 'fit')


def restoreByMouse(event):
    #  Use the right mouse button to reset the image
    global canvas
    global IMG
    IMG = cv2.imread(filename.get())
    Showimage(IMG, canvas, 'fit')
    rotate_scale.set(0)
    enlarge_scale.set(1)
    cutR.set(1)
    cutL.set(0)
    cutD.set(1)
    cutU.set(0)


def restore(event=None):
    #  Reset image and control components
    global canvas
    global IMG
    IMG = cv2.imread(filename.get())
    Showimage(IMG, canvas, 'fit')
    rotate_scale.set(0)
    enlarge_scale.set(1)
    cutR.set(1)
    cutL.set(0)
    cutD.set(1)
    cutU.set(0)


def fileSave(event=None):
    new_name = filedialog.asksaveasfilename(
        defaultextension='.png')  # Set the save file and return the file name, specify the file name suffix as .png
    new_filename.set(new_name)  # Set the value of the variable filename
    # print(str(new_filename.get()))
    temp = img_rotate(IMG, rotate_value.get())  # Rotate first
    sp = temp.shape
    x = int(sp[0] / enlarge_value.get())
    y = int(sp[1] / enlarge_value.get())
    sIMG = temp[sp[0] - x:x, sp[1] - y:y]  # Zoom in again
    cv2.imwrite(new_filename.get(), sIMG)
    showinfo(title="", message="The file is saved successfully")


def img_rotate(image, angle):
    #  Image rotation with image and rotation angle
    (h, w) = image.shape[0:2]
    (cX, cY) = (w // 2, h // 2)
    #  Calculate the transformation matrix, the parameters are represented once (center of rotation, rotation angle,
    #  scaling factor), where the rotation angle is counterclockwise
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    #  Resize the output image
    nH = int(abs(h * math.cos(math.radians(angle))) + abs(w * math.sin(math.radians(angle))))
    nW = int(abs(h * math.sin(math.radians(angle))) + abs(w * math.cos(math.radians(angle))))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))


def rotate_IMG(angle):
    global IMG
    global canvas
    rotate_value.set(int(angle))
    temp = img_rotate(IMG, int(angle))
    value = enlarge_value.get()
    sp = temp.shape
    x = int(sp[0] / value)
    y = int(sp[1] / value)
    sIMG = temp[sp[0] - x:x, sp[1] - y:y]
    Showimage(sIMG, canvas, TYPE=1)


def enlarge_IMG(value):
    global canvas
    global IMG
    value = float(value)
    enlarge_value.set(value)
    temp = img_rotate(IMG, rotate_value.get())
    sp = temp.shape
    x = int(sp[0] / value)
    y = int(sp[1] / value)
    sIMG = temp[sp[0] - x:x, sp[1] - y:y]
    Showimage(sIMG, canvas, TYPE=1)


def openCutWin():
    cut_window = Tk()
    cut_window.title("cut image")
    cut_window.geometry("800x600")
    canvas_cut = tkinter.Canvas(root, width=600, height=450, bg="white")
    canvas_cut.place(x=10, y=10, width=600, height=450)
    temp = cv2.imread(filename.get())
    Showimage(temp, canvas_cut, 'fit')


def cutIMG(value):
    global IMG
    angle = rotate_value.get()
    temp = img_rotate(IMG, angle)
    value = enlarge_value.get()
    sp = temp.shape
    x = int(sp[0] / value)
    y = int(sp[1] / value)
    sIMG = temp[sp[0] - x:x, sp[1] - y:y]
    sp = sIMG.shape
    L = int(sp[1] * cutL.get())
    R = int(sp[1] * cutR.get())
    if R <= L:
        R = L + 1
    U = int(sp[0] * cutU.get())
    D = int(sp[0] * cutD.get())
    if D <= U:
        D = U + 1
    sIMG = sIMG[U:D, L:R]
    Showimage(sIMG, canvas, TYPE=1)


def mixIMG(value):
    global IMG
    global canvas
    global cover
    R1 = mixL.get()
    R2 = mixR.get()
    cover = cv2.resize(cover, (IMG.shape[1], IMG.shape[0]))
    tIMG = cover * R2 + IMG * R1
    np.clip(tIMG, 0, 255, tIMG)
    tIMG = tIMG.astype("uint8")
    Showimage(tIMG, canvas, TYPE=1)


def temp_save():
    """
    Temporarily save the current processing results
    :return:
    """
    global IMG
    global cover
    angle = rotate_value.get()
    temp = img_rotate(IMG, angle)
    value = enlarge_value.get()
    sp = temp.shape
    x = int(sp[0] / value)
    y = int(sp[1] / value)
    sIMG = temp[sp[0] - x:x, sp[1] - y:y]
    sp = sIMG.shape
    L = int(sp[1] * cutL.get())
    R = int(sp[1] * cutR.get())
    U = int(sp[0] * cutU.get())
    D = int(sp[0] * cutD.get())
    IMG = sIMG[U:D, L:R]

    R1 = mixL.get()
    R2 = mixR.get()
    cover = cv2.resize(cover, (IMG.shape[1], IMG.shape[0]))
    tIMG = cover * R2 + IMG * R1
    np.clip(tIMG, 0, 255, tIMG)
    tIMG = tIMG.astype("uint8")
    IMG = tIMG

    tIMG = IMG.astype("float32")
    tIMG[:, :, 0] = IMG[:, :, 0] * (0.5 + RGB_G.get())
    tIMG[:, :, 1] = IMG[:, :, 1] * (0.5 + RGB_B.get())
    tIMG[:, :, 2] = IMG[:, :, 2] * (0.5 + RGB_R.get())
    tIMG[IMG > 255] = 255
    IMG = tIMG.astype("uint8")

    rotate_scale.set(0)
    enlarge_scale.set(1)
    cutR.set(1)
    cutL.set(0)
    cutD.set(1)
    cutU.set(0)
    mixL.set(1)
    mixR.set(0)
    RGB_R.set(0.5)
    RGB_B.set(0.5)
    RGB_G.set(0.5)

    enlarge_scale.pack_forget()
    rotate_scale.pack_forget()
    cut_scaleD.pack_forget()
    cut_scaleL.pack_forget()
    cut_scaleR.pack_forget()
    cut_scaleU.pack_forget()
    mix_scaleL.pack_forget()
    mix_scaleR.pack_forget()
    R_scale.pack_forget()
    G_scale.pack_forget()
    B_scale.pack_forget()
    adjust_mole.set(" ")
    Showimage(tIMG, canvas)


def img_colorless(img, h):
    """
    :param img: Input Image
    :param h: Depigmentation factor, the larger the value the stronger the depigmentation effect
    :return: Return the image after the color removal is completed
    """
    img = np.round(img / h)
    img = img * h
    return img


def img_colorMix(img, k, m, n):
    """
    :param img: input
    :param k: G
    :param m: B
    :param n: R
    :return: Image after the color removal is completed
    """
    temp = img
    temp[:, :, k] = img[:, :, 0]
    temp[:, :, m] = img[:, :, 1]
    temp[:, :, n] = img[:, :, 2]
    return temp


def img_popart():
    """
    :param img: input
    :return: pop art image
    """
    global IMG
    global canvas
    sp = IMG.shape
    temp = img_colorless(IMG, 64)  # Lower the color gradation
    # temp = img_watercolour(img)
    tIMG = np.zeros(shape=[sp[0] * 2 + 30, sp[1] * 3 + 40, 3])  # Large image for display
    tIMG = tIMG.astype("uint8")
    tIMG[10:sp[0] + 10, 10:sp[1] + 10, :] = 125 - img_colorMix(temp, 0, 2, 1)
    tIMG[10:sp[0] + 10, sp[1] + 20:2 * sp[1] + 20, :] = img_colorMix(temp, 0, 1, 2)
    tIMG[10:sp[0] + 10, 2 * sp[1] + 30:3 * sp[1] + 30, :] = img_colorMix(temp, 1, 0, 2)
    tIMG[20 + sp[0]:2 * sp[0] + 20, 10:sp[1] + 10, :] = img_colorMix(temp, 0, 0, 0)
    tIMG[20 + sp[0]:2 * sp[0] + 20, sp[1] + 20:2 * sp[1] + 20, :] = img_colorMix(temp, 0, 2, 2)
    tIMG[20 + sp[0]:2 * sp[0] + 20, 2 * sp[1] + 30:3 * sp[1] + 30, :] = IMG
    IMG = img_adjustSize(tIMG, 1000)
    Showimage(tIMG, canvas)


def img_watercolour(s=100, r=0.3):
    global IMG
    global canvas
    if s > 200:
        s = 200
    if r > 1:
        r = 1
    IMG = cv2.stylization(IMG, sigma_s=s, sigma_r=r)
    Showimage(IMG, canvas)


class OilPaintingThread(threading.Thread):
    def run(self):
        global IMG
        IMG = img_adjustSize(IMG, 300)
        IMG = oilPainting(IMG, 4, 8, 1)
        time_scale.pack_forget()
        P.set(100)
        Showimage(IMG, canvas)
        P.set(0)


def img_oilPainting():
    show_JDT()
    thread1 = OJDThread()
    thread2 = OilPaintingThread()
    thread1.start()
    thread2.start()


def oilPainting(img, templateSize, bucketSize, step):
    """
    :param img: input
    :param templateSize: template size,step
    :param bucketSize: bucket size
    :param step: slide step
    :return: Oil painting effect img
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = ((gray / 256) * bucketSize).astype(int)  # Partitions in the bucket to which the grayscale map belongs
    h, w = img.shape[:2]
    oilImg = np.zeros([h, w, 3], np.uint8)  # Used to store filtered images

    for i in range(0, h, step):
        top = i - templateSize
        bottom = i + templateSize + 1
        if top < 0:
            top = 0
        if bottom >= h:
            bottom = h - 1

        for j in range(0, w, step):

            left = j - templateSize
            right = j + templateSize + 1
            if left < 0:
                left = 0
            if right >= w:
                right = w - 1

            # Grayscale level statistics
            buckets = np.zeros(bucketSize, np.uint8)
            # Bucket array, counting the number of shades of gray in each bucket
            bucketsMean = [0, 0, 0]
            # For the bucket with the most pixels, find the three-channel color mean of all pixels in the bucket
            # Traversal of the template
            for c in range(top, bottom):
                for r in range(left, right):
                    buckets[gray[c, r]] += 1
                    # The pixels within the template are put into the corresponding buckets in turn
                    # somewhat like a grayscale histogram
            maxBucket = np.max(buckets)  # Find the bucket with the most pixels and its index
            maxBucketIndex = np.argmax(buckets)

            for c in range(top, bottom):
                for r in range(left, right):
                    if gray[c, r] == maxBucketIndex:
                        bucketsMean += img[c, r]
            bucketsMean = (bucketsMean / maxBucket).astype(int)  # Three-channel color averaging

            # Oil painting drawings
            for m in range(step):
                for n in range(step):
                    oilImg[m + i, n + j] = (bucketsMean[0], bucketsMean[1], bucketsMean[2])
    return oilImg


def img_pencil():
    global IMG
    global canvas
    IMG[:, :, 1], res = cv2.pencilSketch(IMG, sigma_s=100, sigma_r=0.05, shade_factor=0.1)
    IMG[:, :, 2] = IMG[:, :, 1]
    IMG[:, :, 0] = IMG[:, :, 1]
    Showimage(IMG, canvas)


def img2cartoon(img):
    img_rgb = img
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 1)  # The larger the value, the more blurred it is (taking odd numbers)
    #  Reinforced edge lines
    img_edge = cv2.adaptiveThreshold(img_blur, 128, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=8)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)  # Color image to grayscale image
    img_cartoon = cv2.bitwise_and(img_rgb, img_edge)  # Grayscale image to color image
    #  Adjust brightness and contrast
    res = np.uint8(np.clip((img_cartoon + 64), 0, 255))
    return res


def img2water_ink(img):
    img1 = cv2.medianBlur(img, 7)
    ret, img2 = cv2.threshold(img1, 200, 255, 2, img1)  # Binarization functions
    dilate_kernel = np.ones((3, 3), np.uint8)
    img3 = cv2.dilate(img2, dilate_kernel)
    erode_kernel = np.ones((7, 7), np.uint8)
    img4 = cv2.erode(img3, erode_kernel)
    return img4


def img2ink():
    global IMG
    global canvas
    IMG = img2water_ink(IMG)
    Showimage(IMG, canvas)


def img2ink_gray():
    global IMG
    global canvas
    tIMG = img2water_ink(IMG)
    IMG[:, :, 0] = cv2.cvtColor(tIMG, cv2.COLOR_BGR2GRAY)
    IMG[:, :, 1] = cv2.cvtColor(tIMG, cv2.COLOR_BGR2GRAY)
    IMG[:, :, 2] = cv2.cvtColor(tIMG, cv2.COLOR_BGR2GRAY)
    IMG[IMG > 200] = 255
    Showimage(IMG, canvas)


def CB():
    global IMG
    global canvas
    IMG = img2cartoon(IMG)
    Showimage(img_rotate(IMG, rotate_value.get()), canvas, "fit")  # Cartoon effect button response event


def img_cyberpunk():
    """
    Cyberpunk style filters
    :return:
    """
    global IMG
    image_hls = cv2.cvtColor(IMG, cv2.COLOR_BGR2HLS)
    image_hls = np.asarray(image_hls, np.float32)
    hue = image_hls[:, :, 0]
    hue[hue < 90] = 180 - hue[hue < 90]
    image_hls[:, :, 0] = hue
    image_hls = np.asarray(image_hls, np.uint8)
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2BGR)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image_lab = np.asarray(image_lab, np.float32)
    light_gamma_high = np.power(image_lab[:, :, 0], 0.8)
    light_gamma_high = np.asarray(light_gamma_high / np.max(light_gamma_high) * 255, np.uint8)
    light_gamma_low = np.power(image_lab[:, :, 0], 1.2)
    light_gamma_low = np.asarray(light_gamma_low / np.max(light_gamma_low) * 255, np.uint8)
    dark_b = image_lab[:, :, 2] * (light_gamma_low / 255) * 0.1
    dark_a = image_lab[:, :, 2] * (1 - light_gamma_high / 255) * 0.3
    image_lab[:, :, 2] = np.clip(image_lab[:, :, 2] - dark_b, 0, 255)
    image_lab[:, :, 2] = np.clip(image_lab[:, :, 2] - dark_a, 0, 255)
    image_lab = np.asarray(image_lab, np.uint8)
    IMG = cv2.cvtColor(image_lab, cv2.COLOR_Lab2BGR)
    Showimage(IMG, canvas)


def img_old_photo():
    global IMG
    img = IMG
    rows, cols = IMG.shape[:2]
    dst = np.zeros((rows, cols, 3), dtype="float32")
    dst[:, :, 0] = 0.272 * img[:, :, 2] + 0.534 * img[:, :, 1] + 0.131 * img[:, :, 0]
    dst[:, :, 1] = 0.349 * img[:, :, 2] + 0.686 * img[:, :, 1] + 0.168 * img[:, :, 0]
    dst[:, :, 2] = 0.393 * img[:, :, 2] + 0.769 * img[:, :, 1] + 0.189 * img[:, :, 0]
    dst[dst > 255] = 255
    IMG = dst.astype("uint8")
    Showimage(IMG, canvas)


def Portrait_beautification(value):
    global IMG
    image_dst = cv2.bilateralFilter(IMG, value, value * 2, value / 2)
    IMG = image_dst
    Showimage(IMG, canvas)


def Filter(*arg):
    filter_type = box_value.get()
    if filter_type == "Cartoon":
        CB()
    elif filter_type == "Popart":
        img_popart()
    elif filter_type == "WaterColor":
        img_watercolour()
    elif filter_type == "OilPainting":
        img_oilPainting()
    elif filter_type == "PencilSketch":
        img_pencil()


def img_slice():
    global IMG
    global canvas
    r1 = random.random()
    r2 = 1
    r3 = random.random()
    h, w, k = IMG.shape
    timg = np.zeros(IMG.shape) + 255

    timg[int(h / 6) + 2:int(h / 3) - 2, 0:int(r1 * w), :] = IMG[int(h / 6) + 2:int(h / 3) - 2, (w - int(r1 * w)):w, :]
    timg[int(h / 6) + 2:int(h / 3) - 2, int(r1 * w):w, :] = IMG[int(h / 6) + 2:int(h / 3) - 2, 0:(w - int(r1 * w)), :]

    timg[int(h / 3) + 4:2 * int(h / 3), 0:int(r2 * w), :] = IMG[int(h / 3) + 4:2 * int(h / 3), (w - int(r2 * w)):w, :]
    timg[int(h / 3) + 4:2 * int(h / 3), int(r2 * w):w, :] = IMG[int(h / 3) + 4:2 * int(h / 3), 0:(w - int(r2 * w)), :]

    timg[2 * int(h / 3) + 3:5 * int(h / 6) - 2, 0:int(r3 * w), :] = IMG[2 * int(h / 3) + 3:5 * int(h / 6) - 2,
                                                                    (w - int(r3 * w)):w, :]
    timg[2 * int(h / 3) + 3:5 * int(h / 6) - 2, int(r3 * w):w, :] = IMG[2 * int(h / 3) + 3:5 * int(h / 6) - 2,
                                                                    0:(w - int(r3 * w)), :]

    timg[0:int(h / 6), :, :] = IMG[0:int(h / 6), :, :]
    timg[5 * int(h / 6):h, :, :] = IMG[5 * int(h / 6):h, :, :]

    IMG = timg.astype("uint8")
    Showimage(IMG, canvas)


def placeItem(name):
    global canvas
    enlarge_scale.pack_forget()
    rotate_scale.pack_forget()
    cut_scaleD.pack_forget()
    cut_scaleL.pack_forget()
    cut_scaleR.pack_forget()
    cut_scaleU.pack_forget()
    R_scale.pack_forget()
    G_scale.pack_forget()
    B_scale.pack_forget()
    mix_scaleL.place_forget()
    mix_scaleR.place_forget()

    global tip_flag
    if name == "Enlarge":
        enlarge_scale.pack(fill="x", padx=30, pady=20)
    elif name == "Rotate":
        rotate_scale.pack(fill="x", padx=30, pady=10)
    elif name == "Cut":
        cut_scaleL.pack(side="bottom", fill="x", anchor="s", padx=20, pady=5)
        cut_scaleR.pack(side="bottom", fill="x", anchor="s", padx=20, pady=5)
        cut_scaleU.pack(side="right", fill="y", padx=10, pady=5)
        cut_scaleD.pack(side="right", fill="y", padx=10, pady=5)

    elif name == "RGB":
        R_scale.pack(fill="x")
        G_scale.pack(fill="x")
        B_scale.pack(fill="x")

    elif name == "Mix":
        global cover
        cover_name = filedialog.askopenfilename()
        cover = cv2.imread(cover_name)
        cover = cv2.resize(cover, (IMG.shape[1], IMG.shape[0]))
        mix_scaleL.place(x=50, y=530, width=700, height=20)
        mix_scaleR.place(x=50, y=550, width=700, height=20)

    elif name == "Brightness":
        adjust_mole.set("Brightness")
        if tip_flag == 1:
            showinfo(title="tips", message="Use the mouse wheel to make color adjustments!")
            tip_flag = 0

    elif name == "Saturation":
        adjust_mole.set("Saturation")
        if tip_flag == 1:
            showinfo(title="tips", message="Use the mouse wheel to make color adjustments!")
            tip_flag = 0

    elif name == "Contrast":
        adjust_mole.set("Contrast")
        if tip_flag == 1:
            showinfo(title="tips", message="Use the mouse wheel to make color adjustments!")
            tip_flag = 0

    elif name == "Shadow_improvement":
        adjust_mole.set("Shadow_improvement")
        if tip_flag == 1:
            showinfo(title="tips", message="Use the mouse wheel to make color adjustments!")
            tip_flag = 0

    elif name == "Highlight":
        adjust_mole.set("Highlight")
        if tip_flag == 1:
            showinfo(title="tips", message="Use the mouse wheel to make color adjustments!")
            tip_flag = 0


def RGB_edit(value):
    global IMG
    tIMG = IMG.astype("float32")
    tIMG[:, :, 0] = IMG[:, :, 0] * (0.5 + RGB_R.get())
    tIMG[:, :, 1] = IMG[:, :, 1] * (0.5 + RGB_G.get())
    tIMG[:, :, 2] = IMG[:, :, 2] * (0.5 + RGB_B.get())
    tIMG[tIMG > 255] = 255
    tIMG = tIMG.astype("uint8")
    Showimage(tIMG, canvas, TYPE=1)


def img_auto_colorful():
    global IMG
    med = np.median(IMG.astype(np.float32))  # Get the median M
    img_temp = 1 / (1 + np.power((med / (IMG + 1e-6)), 4.5))  # 4.5 is the slope
    temp2 = img_temp - np.min(img_temp)
    img_con_str = np.uint8(255 * (temp2 / np.max(temp2)))
    IMG = img_con_str
    Showimage(IMG, canvas)


def img_BrightnessEdit(l):
    if l > 0:
        l = 1.05
    else:
        l = 0.94
    global IMG
    sp = IMG.shape
    tIMG = IMG * l
    tIMG[tIMG > 255] = 255
    # hsv = cv2.cvtColor(IMG, cv2.COLOR_BGR2HSV)
    # hsv[:, :, 2] = hsv[:, :, 2] * l
    # np.clip(hsv, 0, 255, hsv)
    # IMG = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # np.clip(IMG, 0, 255, IMG)
    tIMG = tIMG.astype("uint8")
    IMG = tIMG
    Showimage(tIMG, canvas)


def img_SaturationEdit(l):
    global IMG
    if l > 0:
        l = 1.05
    else:
        l = 0.97
    tIMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2HLS)
    tIMG = tIMG.astype("float32")
    # The data type conversion is important here because unsigned 8-bit shaping will overflow
    tIMG[:, :, 2] = tIMG[:, :, 2] * l
    tIMG[:, :, 2][tIMG[:, :, 2] > 255] = 255
    tIMG = tIMG.astype("uint8")
    IMG = cv2.cvtColor(tIMG, cv2.COLOR_HLS2BGR)
    Showimage(IMG, canvas)


def img_ShadowEdit(l):
    global IMG
    if l > 0:
        l = 1.03
    else:
        l = 0.97
    sp = IMG.shape
    pixels = sp[0] * sp[1]
    tIMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2HLS)
    tIMG = tIMG.astype("float32")
    mid = sum(sum(tIMG[:, :, 1])) / pixels - 20
    tIMG[:, :, 1][tIMG[:, :, 1] < mid] = tIMG[:, :, 1][tIMG[:, :, 1] < mid] * l
    tIMG = tIMG.astype("uint8")
    IMG = cv2.cvtColor(tIMG, cv2.COLOR_HLS2BGR)
    Showimage(IMG, canvas)


def img_HighlightEdit(l):
    global IMG
    if l > 0:
        l = 1.03
    else:
        l = 0.97
    sp = IMG.shape
    pixels = sp[0] * sp[1]
    tIMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2HLS)
    tIMG = tIMG.astype("float32")
    mid = sum(sum(tIMG[:, :, 1])) / pixels + 20
    tIMG[:, :, 1][tIMG[:, :, 1] > mid] = tIMG[:, :, 1][tIMG[:, :, 1] > mid] * l
    tIMG[:, :, 1][tIMG[:, :, 1] > 255] = 255
    tIMG = tIMG.astype("uint8")
    IMG = cv2.cvtColor(tIMG, cv2.COLOR_HLS2BGR)
    Showimage(IMG, canvas)


def img_ContrastEdit(l):
    if l > 0:
        l = 0.05
    else:
        l = -0.05
    global IMG
    sp = IMG.shape
    pixels = sp[0] * sp[1]
    r = np.sum(IMG[:, :, 0]) / pixels
    g = np.sum(IMG[:, :, 1]) / pixels
    b = np.sum(IMG[:, :, 2]) / pixels
    tIMG = IMG.astype("float32")
    tIMG[:, :, 0] = IMG[:, :, 0] + (IMG[:, :, 0] - r) * l
    tIMG[:, :, 1] = IMG[:, :, 1] + (IMG[:, :, 1] - g) * l
    tIMG[:, :, 2] = IMG[:, :, 2] + (IMG[:, :, 2] - b) * l
    tIMG[tIMG > 255] = 255
    tIMG[tIMG < 0] = 0
    IMG = tIMG
    IMG = IMG.astype("uint8")
    Showimage(IMG, canvas)


def place_temp_save_button(event):
    TempSave.place(x=event.x, y=event.y, width=60, height=20)
    restore_button.place(x=event.x, y=event.y + 20, width=60, height=20)


def hide_temp_save_button(event):
    TempSave.place_forget()
    restore_button.place_forget()


def img_adjustSize(image, num):
    x, y = image.shape[0:2]
    i = 1
    if x * y > num * num:
        i = (num / x + num / y) / 2
    image = cv2.resize(image, (int(y * i), int(x * i)))
    return image


def adjustByMouse(event):
    if adjust_mole.get() == "Brightness":
        img_BrightnessEdit(event.delta)
    elif adjust_mole.get() == "Saturation":
        img_SaturationEdit(event.delta)
    elif adjust_mole.get() == "Contrast":
        img_ContrastEdit(event.delta)
    elif adjust_mole.get() == "Shadow_improvement":
        img_ShadowEdit(event.delta)
    elif adjust_mole.get() == "Highlight":
        img_HighlightEdit(event.delta)


def img_color_transfer():
    global IMG
    target = IMG
    t_name = filedialog.askopenfilename()  # Get the selected file
    source = cv2.imread(t_name)
    # Conversion from RGB space to LAB space
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    # Conversion from RGB space to LAB space
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # Divide the L, a and b channels of the target image and subtract the corresponding mean values
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    # Standardization of L, a and b channels respectively
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b

    # Plus the mean value
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # Restrict the result of processing to the space of [0,255]
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # Combine the L, a, and b channels and convert them back to RGB color space
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    IMG = transfer
    Showimage(IMG, canvas)


def image_stats(image):
    # Calculate the mean and variance values for each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())
    # Return the corresponding statistical information
    return lMean, lStd, aMean, aStd, bMean, bStd


def img_style_transform(model):
    global IMG
    IMG = img_adjustSize(IMG, 400)
    net = cv2.dnn.readNetFromTorch(r"models\candy.t7")
    if model == "candy":
        net = cv2.dnn.readNetFromTorch(r"models\candy.t7")
    elif model == "composition":
        net = cv2.dnn.readNetFromTorch(r"models\composition_vii.t7")
    elif model == "feathers":
        net = cv2.dnn.readNetFromTorch(r"models\feathers.t7")
    elif model == "la_muse":
        net = cv2.dnn.readNetFromTorch(r"models\la_muse.t7")
    elif model == "mosaic":
        net = cv2.dnn.readNetFromTorch(r"models\mosaic.t7")
    elif model == "starry_night":
        net = cv2.dnn.readNetFromTorch(r"models\starry_night.t7")
    elif model == "scream":
        net = cv2.dnn.readNetFromTorch(r"models\the_scream.t7")
    elif model == "wave":
        net = cv2.dnn.readNetFromTorch(r"models\the_wave.t7")
    elif model == "udnie":
        net = cv2.dnn.readNetFromTorch(r"models\udnie.t7")

    cap = IMG
    sp = IMG.shape
    pixels = sp[0] * sp[1]
    r = np.sum(IMG[:, :, 0]) / pixels
    g = np.sum(IMG[:, :, 1]) / pixels
    b = np.sum(IMG[:, :, 2]) / pixels

    inp = cv2.dnn.blobFromImage(cap, 1.0, (cap.shape[1], cap.shape[0]), (r, g, b), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0] += r
    out[1] += g
    out[2] += b
    out = out.transpose(1, 2, 0)
    out[out > 255] = 255
    out[out < 0] = 0
    IMG = out
    IMG = IMG.astype("uint8")
    Showimage(IMG, canvas)


def show_JDT():
    time_scale.pack(anchor="center", expand="True", ipadx=10, ipady=10, fill="x")


def img_to_pix():
    show_JDT()
    thread1 = JDThread()
    thread2 = PIXThread()
    thread1.start()
    thread2.start()


def img_to_lego():
    show_JDT()
    thread1 = JDThread()
    thread2 = LegoThread()
    thread1.start()
    thread2.start()


def img_to_heart():
    show_JDT()
    thread1 = JDThread()
    thread2 = HeartThread()
    thread1.start()
    thread2.start()


def img_to_mc():
    show_JDT()
    thread1 = JDThread()
    thread2 = MCThread()
    thread1.start()
    thread2.start()


def img_glsg():
    show_JDT()
    thread1 = JDThread()
    thread2 = GLSGThread()
    thread1.start()
    thread2.start()


class JDThread(threading.Thread):  # Inherits parent class threading.
    def run(self):
        print("Starting " + self.name)
        while P.get() < 99:
            P.set(P.get() + 0.3)
            time.sleep(0.01)


class OJDThread(threading.Thread):  # Inherits parent class threading.
    def run(self):
        print("Starting " + self.name)
        while P.get() < 99:
            P.set(P.get() + 0.1)
            time.sleep(0.01)


class PIXThread(threading.Thread):
    def run(self):
        global IMG
        tiler.main(filename.get(), r"tiles\circles\gen_circle_100")
        IMG = cv2.imread("out.png")
        time_scale.pack_forget()
        P.set(100)
        Showimage(IMG, canvas)
        P.set(0)


class LegoThread(threading.Thread):
    def run(self):
        global IMG
        tiler.main(filename.get(), r"tiles\lego\gen_lego_v")
        IMG = cv2.imread("out.png")
        time_scale.pack_forget()
        P.set(100)
        Showimage(IMG, canvas)
        P.set(0)


class HeartThread(threading.Thread):
    def run(self):
        global IMG
        tiler.main(filename.get(), r"tiles\hearts\gen_heart")
        IMG = cv2.imread("out.png")
        time_scale.pack_forget()
        P.set(100)
        Showimage(IMG, canvas)
        P.set(0)


class GLSGThread(threading.Thread):
    def run(self):
        global IMG
        tiler.main(filename.get(), r"tiles\gen_glsg")
        IMG = cv2.imread("out.png")
        time_scale.pack_forget()
        P.set(100)
        Showimage(IMG, canvas)
        P.set(0)


class MCThread(threading.Thread):
    def run(self):
        global IMG
        tiler.main(filename.get(), r"tiles\minecraft")
        IMG = cv2.imread("out.png")
        time_scale.pack_forget()
        P.set(100)
        Showimage(IMG, canvas)
        P.set(0)


def img_style_transform_old(model):
    global IMG
    if model == "candy":
        net = cv2.dnn.readNetFromTorch(r"models\candy.t7")
    elif model == "composition":
        net = cv2.dnn.readNetFromTorch(r"models\composition_vii.t7")
    elif model == "feathers":
        net = cv2.dnn.readNetFromTorch(r"models\feathers.t7")
    elif model == "la_muse":
        net = cv2.dnn.readNetFromTorch(r"models\la_muse.t7")
    elif model == "mosaic":
        net = cv2.dnn.readNetFromTorch(r"models\mosaic.t7")
    elif model == "starry_night":
        net = cv2.dnn.readNetFromTorch(r"models\starry_night.t7")
    elif model == "scream":
        net = cv2.dnn.readNetFromTorch(r"models\the_scream.t7")
    elif model == "wave":
        net = cv2.dnn.readNetFromTorch(r"models\the_wave.t7")
    elif model == "udnie":
        net = cv2.dnn.readNetFromTorch(r"models\udnie.t7")
    cap = IMG
    sp = IMG.shape
    pixels = sp[0] * sp[1]
    r = np.sum(IMG[:, :, 0]) / pixels
    g = np.sum(IMG[:, :, 1]) / pixels
    b = np.sum(IMG[:, :, 2]) / pixels

    inp = cv2.dnn.blobFromImage(cap, 1.0, (cap.shape[1], cap.shape[0]), (r, g, b), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0] += r
    out[1] += g
    out[2] += b
    out = out.transpose(1, 2, 0)
    out[out > 255] = 255
    out[out < 0] = 0
    IMG = out
    IMG = IMG.astype("uint8")
    Showimage(IMG, canvas)


def img_remove(img):
    img = Image.fromarray(img)
    img = remove(img)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def img_get_foreground(type):
    global IMG
    IMG = img_remove(IMG)
    black_pixels = IMG[:, :, 0] + IMG[:, :, 1] + IMG[:, :, 2] < 3
    if type == "W":
        IMG[black_pixels] = [255, 255, 255]
    if type == "R":
        IMG[black_pixels] = [0, 0, 255]
    if type == "B":
        IMG[black_pixels] = [255, 0, 0]
    Showimage(IMG, canvas)


def Segment(src, sky_img):
    """
    Change the sky
    :param src: input
    :param sky_img: new background
    :return:
    """
    hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    hsv_img[2] = cv2.equalizeHist(hsv_img[2])

    lower_red = np.array([100, 48, 40])
    upper_red = np.array([124, 255, 255])
    range_img = cv2.inRange(hsv_img, lower_red, upper_red)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    Erode_img = cv2.erode(range_img, kernel)
    Dilation_img = cv2.dilate(Erode_img, kernel)

    ret, mask = cv2.threshold(Dilation_img, 150, 255, cv2.THRESH_BINARY)
    not_mask = cv2.bitwise_not(mask)
    front_pic = cv2.bitwise_and(src, src, mask=not_mask)

    sky_resize = cv2.resize(sky_img, (src.shape[1], src.shape[0]))
    back_image = cv2.bitwise_and(sky_resize, sky_resize, mask=mask)
    merge_img = cv2.add(back_image, front_pic)
    return merge_img


def change_sky():
    global IMG
    sky = cv2.imread(filedialog.askopenfilename())
    IMG = Segment(IMG, sky)
    Showimage(IMG, canvas)


def illumination(img, lightIntensity, x=0, y=0):  # Parameters are original image and light intensity
    h, w = img.shape[0:2]
    img1 = np.zeros((h, w, 3), dtype=img.dtype)
    x, y = int(h / 2), int(w / 2)
    r = min(x, y)
    for i in range(h):
        for j in range(w):
            distance = (x - i) ** 2 + (y - j) ** 2
            if distance > r ** 2:
                img1[i, j] = img[i, j]
            else:
                result = int(lightIntensity * (1.0 - np.sqrt(distance) / r))
                B = min(max(0, img[i, j, 0] + result), 255)
                G = min(max(0, img[i, j, 1] + result), 255)
                R = min(max(0, img[i, j, 2] + result), 255)
                img1[i, j] = [B, G, R]
    return img1


def img_lightning():
    global IMG
    IMG = illumination(IMG, 150)
    Showimage(IMG, canvas)


def imgAddBorder(img, img2):
    img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(img, img, mask=mask)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
    dst = cv2.add(img1_bg, img2_fg)
    return dst


def img_border(type):
    global IMG
    border = cv2.imread("edge/e" + str(type) + ".jpg")
    IMG = imgAddBorder(IMG, border)
    Showimage(IMG, canvas)


def fastFlash(img):
    global IMGs
    global position
    global count
    N = len(IMGs)
    for i in range(0, N - position):
        IMGs.pop(-1)
    IMGs.append(img)
    position = position + 1


def Undo(event=None):
    global IMGs
    global IMG
    global position
    if position > 1:
        position = position - 1
        IMG = IMGs[position - 1]
    Showimage(IMG, canvas, TYPE=1)
    print(position)


def Redo(event=None):
    global IMGs
    global IMG
    global position
    if position < len(IMGs):
        position = position + 1
        IMG = IMGs[position - 1]
    Showimage(IMG, canvas, TYPE=1)
    print(position)


def Quit(event=None):
    if askyesno('Warning!', "Confirm exit? Your operation may not have been saved!"):
        root.quit()


if __name__ == '__main__':
    style = Style(theme="lumen")  # style = Style(theme="lumen")
    root = style.master
    root.title("ImageCraft")
    root.geometry("800x630")

    global IMG
    global IMGs
    global position
    global canvas
    global cover

    filename = tkinter.StringVar()
    # tk is the method used to record the name of the variable, here the file name of the image
    new_filename = tkinter.StringVar()
    # File name for saving
    adjust_mole = tkinter.StringVar()
    # Mode of color adjustment
    box_value = tkinter.StringVar()
    # Drop-down box for filter options
    rotate_value = IntVar()
    # Image rotation value
    enlarge_value = DoubleVar()
    # Image magnification value
    cutR = tkinter.DoubleVar()  # Right
    cutL = tkinter.DoubleVar()  # Left
    cutU = tkinter.DoubleVar()  # UP
    cutD = tkinter.DoubleVar()  # Down
    mixL = tkinter.DoubleVar()
    mixR = tkinter.DoubleVar()

    RGB_R = tkinter.DoubleVar()  # RGB
    RGB_G = tkinter.DoubleVar()
    RGB_B = tkinter.DoubleVar()

    P = tkinter.DoubleVar()  # Progress bar

    rotate_value.set(0)  # Initialize to 0
    enlarge_value.set(1)
    cutR.set(1)
    cutL.set(0)
    cutU.set(0)
    cutD.set(1)
    mixL.set(1)
    mixR.set(0)
    RGB_R.set(0.5)
    RGB_B.set(0.5)
    RGB_G.set(0.5)

    # Enlarge slider
    enlarge_scale = Scale(root, from_=1, to=1.85, resolution=0.01, background="red", bd=1, variable=enlarge_value,
                          command=enlarge_IMG, showvalue=False, orient=HORIZONTAL)
    # Rotating slider
    rotate_scale = Scale(root, from_=-180, to=180, resolution=1, showvalue=False, background="red", orient=HORIZONTAL,
                         variable=rotate_value, command=rotate_IMG)
    # Bind the slider's value to the global image rotation value

    # Cutting sliders
    cut_scaleL = Scale(root, from_=0, to=1, resolution=0.001, showvalue=False, orient=HORIZONTAL, variable=cutL,
                       command=cutIMG)  # Cut from the left
    cut_scaleR = Scale(root, from_=0, to=1, resolution=0.001, showvalue=False, orient=HORIZONTAL, variable=cutR,
                       command=cutIMG)  # Cut from the right
    cut_scaleU = Scale(root, from_=0, to=1, resolution=0.001, showvalue=False, variable=cutU, command=cutIMG)
    # Cut from the left
    cut_scaleD = Scale(root, from_=0, to=1, resolution=0.001, showvalue=False, variable=cutD, command=cutIMG)
    # Cut from the left

    # Hybrid slider
    mix_scaleL = Scale(root, from_=0, to=1, resolution=0.001, showvalue=False, orient=HORIZONTAL, variable=mixL,
                       command=mixIMG)
    mix_scaleR = Scale(root, from_=0, to=1, resolution=0.001, showvalue=False, orient=HORIZONTAL, variable=mixR,
                       command=mixIMG)

    R_scale = Scale(root, from_=0.2, to=0.8, resolution=0.001, showvalue=False, orient=HORIZONTAL, variable=RGB_R,
                    command=RGB_edit)
    G_scale = Scale(root, from_=0.2, to=0.8, resolution=0.001, showvalue=False, orient=HORIZONTAL, variable=RGB_G,
                    command=RGB_edit)
    B_scale = Scale(root, from_=0.2, to=0.8, resolution=0.001, showvalue=False, orient=HORIZONTAL, variable=RGB_B,
                    command=RGB_edit)

    # Mouse event binding
    root.bind("<Button-3>", place_temp_save_button)  # Right mouse button
    root.bind("<ButtonRelease-1>", hide_temp_save_button)  # Stand-alone release
    root.bind("<MouseWheel>", adjustByMouse)  # Mouse events
    root.bind("<Double-Button-1>", restoreByMouse)
    root.bind_all("<Control-z>", Undo)  # rollback
    root.bind_all("<Control-Z>", Undo)
    root.bind_all("<Control-y>", Redo)  # Redo
    root.bind_all("<Control-Y>", Redo)
    root.bind_all("<Control-s>", fileSave)  # save
    root.bind_all("<Control-S>", fileSave)
    root.bind_all("<Control-O>", findIMG)  # open
    root.bind_all("<Control-o>", findIMG)
    root.bind_all("<Control-Q>", Quit)  # quit
    root.bind_all("<Control-q>", Quit)
    root.bind_all("<Control-R>", restore)
    root.bind_all("<Control-r>", restore)

    # File menu options
    menubar = Menu(root, tearoff=False)
    root.config(menu=menubar)
    file_menu = tkinter.Menu(menubar, tearoff=False)
    for item in ["Open", "Restart", "Edit", "Save", "Quit"]:
        if item == "Quit":
            file_menu.add_separator()  # 分割线
            file_menu.add_command(label=item, command=root.quit)
        elif item == "Open":
            file_menu.add_command(label=item, command=findIMG)
        elif item == "Save":
            file_menu.add_command(label=item, command=fileSave)
        elif item == "Restart":
            file_menu.add_command(label=item, command=lambda:restore(None))
    menubar.add_cascade(label="File", menu=file_menu)

    # Edit Menu Options
    edit_menu = tkinter.Menu(menubar, tearoff=False)
    for item in ["Enlarge", "Rotate", "Cut", "Brightness", "Saturation", "Contrast", "Shadow", "Highlight", "RGB"]:
        if item == "Enlarge":
            edit_menu.add_command(label=item, command=lambda: placeItem("Enlarge"))
        elif item == "Rotate":
            edit_menu.add_command(label=item, command=lambda: placeItem("Rotate"))
        elif item == "Cut":
            edit_menu.add_command(label=item, command=lambda: placeItem("Cut"))
            edit_menu.add_separator()
        elif item == "Brightness":
            edit_menu.add_command(label=item, command=lambda: placeItem("Brightness"))
        elif item == "Saturation":
            edit_menu.add_command(label=item, command=lambda: placeItem("Saturation"))
        elif item == "Contrast":
            edit_menu.add_command(label=item, command=lambda: placeItem("Contrast"))
        elif item == "Shadow":
            edit_menu.add_command(label=item, command=lambda: placeItem("Shadow_improvement"))
        elif item == "Highlight":
            edit_menu.add_command(label=item, command=lambda: placeItem("Highlight"))
        elif item == "RGB":
            edit_menu.add_command(label=item, command=lambda: placeItem("RGB"))
    menubar.add_cascade(label="Edit", menu=edit_menu)

    # Filter menu options
    filter_menu = tkinter.Menu(menubar, tearoff=False)
    for item in ["Cartoon", "InkPainting", "WaterColour", "OilPainting", "PencilSketch", "Cyberpunk",
                 "OldPhoto", "Popart"]:
        if item == "Cartoon":
            filter_menu.add_command(label=item, command=CB)
        elif item == "InkPainting":
            Ink_menu = tkinter.Menu(root)
            Ink_menu.add_command(label=item + "_RGB", command=img2ink)
            Ink_menu.add_command(label=item + "_GRAY", command=img2ink_gray)
            filter_menu.add_cascade(label=item, menu=Ink_menu)
        elif item == "WaterColour":
            filter_menu.add_command(label=item, command=img_watercolour)
        elif item == "OilPainting":
            filter_menu.add_command(label=item, command=img_oilPainting)
        elif item == "PencilSketch":
            filter_menu.add_command(label=item, command=img_pencil)
        elif item == "Cyberpunk":
            filter_menu.add_command(label=item, command=img_cyberpunk)
        elif item == "Popart":
            filter_menu.add_command(label=item, command=img_popart)
        elif item == "OldPhoto":
            filter_menu.add_command(label=item, command=img_old_photo)
    menubar.add_cascade(label="Filter", menu=filter_menu)

    # Beauty Menu
    bilateralFilter_menu = tkinter.Menu(menubar, tearoff=False)
    for item in ["Slight", "Moderate", "High"]:
        if item == "Slight":
            bilateralFilter_menu.add_command(label=item, command=lambda: Portrait_beautification(10))
        elif item == "Moderate":
            bilateralFilter_menu.add_command(label=item, command=lambda: Portrait_beautification(30))
        elif item == "High":
            bilateralFilter_menu.add_command(label=item, command=lambda: Portrait_beautification(60))
    menubar.add_cascade(label="P&B", menu=bilateralFilter_menu)

    # Image borders
    Border_menu = tkinter.Menu(menubar, tearoff=False)
    for item in ["type1", "type2", "type3", "type4", "type5", "type6"]:
        if item == "type1":
            Border_menu.add_command(label=item, command=lambda: img_border(1))
        elif item == "type2":
            Border_menu.add_command(label=item, command=lambda: img_border(3))
        elif item == "type3":
            Border_menu.add_command(label=item, command=lambda: img_border(2))
        elif item == "type4":
            Border_menu.add_command(label=item, command=lambda: img_border(4))
        elif item == "type5":
            Border_menu.add_command(label=item, command=lambda: img_border(5))
        elif item == "type6":
            Border_menu.add_command(label=item, command=lambda: img_border(6))
    menubar.add_cascade(label="Border", menu=Border_menu)

    # Featured Tools
    tool_menu = tkinter.Menu(menubar, tearoff=False)
    # d = {"Mix":lambda: placeItem("Mix"), "Slice", "Colorful", "ColorTransform", "Pixelate", "Lego", "MineCraft"}
    # d = {"Mix": lambda: placeItem("Mix")}
    # for i in d:
    #     tool_menu.add_command(label=i, command=d[i])
    for item in ["Mix", "Slice", "illumination", "Colourful", "Sky_Trans", "GetForeground", "ColourTrans", "Pixelate",
                 "Lego", "Hearts", "MineCraft",
                 "Glasgow"]:
        if item == "Mix":
            tool_menu.add_command(label=item, command=lambda: placeItem("Mix"))
        elif item == "Slice":
            tool_menu.add_command(label=item, command=img_slice)
        elif item == "illumination":
            tool_menu.add_command(label=item, command=img_lightning)
        elif item == "Colourful":
            tool_menu.add_command(label=item, command=img_auto_colorful)
        elif item == "ColourTrans":
            tool_menu.add_command(label=item, command=img_color_transfer)
            tool_menu.add_separator()
        elif item == "GetForeground":
            GF_menu = tkinter.Menu(root)
            GF_menu.add_command(label="Black", command=lambda: img_get_foreground("Black"))
            GF_menu.add_command(label="Withe", command=lambda: img_get_foreground("W"))
            GF_menu.add_command(label="Red", command=lambda: img_get_foreground("R"))
            GF_menu.add_command(label="Blue", command=lambda: img_get_foreground("B"))
            tool_menu.add_cascade(label=item, menu=GF_menu)
        elif item == "Sky_Trans":
            tool_menu.add_command(label=item, command=change_sky)
            tool_menu.add_separator()
        elif item == "Pixelate":
            tool_menu.add_command(label=item, command=img_to_pix)
        elif item == "Lego":
            tool_menu.add_command(label=item, command=img_to_lego)
        elif item == "Hearts":
            tool_menu.add_command(label=item, command=img_to_heart)
        elif item == "MineCraft":
            tool_menu.add_command(label=item, command=img_to_mc)
        elif item == "Glasgow":
            tool_menu.add_command(label=item, command=img_glsg)

    menubar.add_cascade(label="SpecialTool", menu=tool_menu)

    #  Style Migration Tool
    style_menu = tkinter.Menu(menubar, tearoff=False)
    for item in ["Candy", "Feathers", "La_muse", "Mosaic", "Starry_night", "Scream", "Wave", "Udnie", "Composition"]:
        if item == "Candy":
            candy_menu = tkinter.Menu(root)
            candy_menu.add_command(label=item + "_old", command=lambda: img_style_transform("candy"))
            candy_menu.add_command(label=item + "_new", command=lambda: img_style_transform_old("candy"))
            style_menu.add_cascade(label=item, menu=candy_menu)

        elif item == "Composition":
            composition_menu = tkinter.Menu(root)
            composition_menu.add_command(label=item + "_old", command=lambda: img_style_transform("composition"))
            composition_menu.add_command(label=item + "_new", command=lambda: img_style_transform_old("composition"))
            style_menu.add_cascade(label=item, menu=composition_menu)

        elif item == "Feathers":
            feathers_menu = tkinter.Menu(root)
            feathers_menu.add_command(label=item + "_old", command=lambda: img_style_transform("feathers"))
            feathers_menu.add_command(label=item + "_new", command=lambda: img_style_transform_old("feathers"))
            style_menu.add_cascade(label=item, menu=feathers_menu)

        elif item == "La_muse":
            la_muse_menu = tkinter.Menu(root)
            la_muse_menu.add_command(label=item + "_old", command=lambda: img_style_transform("la_muse"))
            la_muse_menu.add_command(label=item + "_new", command=lambda: img_style_transform_old("la_muse"))
            style_menu.add_cascade(label=item, menu=la_muse_menu)

        elif item == "Mosaic":
            mosaic_menu = tkinter.Menu(root)
            mosaic_menu.add_command(label=item + "_old", command=lambda: img_style_transform("mosaic"))
            mosaic_menu.add_command(label=item + "_new", command=lambda: img_style_transform_old("mosaic"))
            style_menu.add_cascade(label=item, menu=mosaic_menu)

        elif item == "Starry_night":
            starry_night_menu = tkinter.Menu(root)
            starry_night_menu.add_command(label=item + "_old", command=lambda: img_style_transform("starry_night"))
            starry_night_menu.add_command(label=item + "_new", command=lambda: img_style_transform_old("starry_night"))
            style_menu.add_cascade(label=item, menu=starry_night_menu)

        elif item == "Scream":
            scream_menu = tkinter.Menu(root)
            scream_menu.add_command(label=item + "_old", command=lambda: img_style_transform("scream"))
            scream_menu.add_command(label=item + "_new", command=lambda: img_style_transform_old("scream"))
            style_menu.add_cascade(label=item, menu=scream_menu)

        elif item == "Udnie":
            udnie_menu = tkinter.Menu(root)
            udnie_menu.add_command(label=item + "_old", command=lambda: img_style_transform("udnie"))
            udnie_menu.add_command(label=item + "_new", command=lambda: img_style_transform_old("udnie"))
            style_menu.add_cascade(label=item, menu=udnie_menu)

        elif item == "Wave":
            wave_menu = tkinter.Menu(root)
            wave_menu.add_command(label=item + "_old", command=lambda: img_style_transform("wave"))
            wave_menu.add_command(label=item + "_new", command=lambda: img_style_transform_old("wave"))
            style_menu.add_cascade(label=item, menu=wave_menu)

    menubar.add_cascade(label="StyleTransform", menu=style_menu)

    # canvas
    canvas = tkinter.Canvas(root, width=700, height=500, bg="white")

    # progress bar
    time_scale = Progressbar(root, length=100, variable=P)

    # temporary storage
    TempSave = Button(root, text="done", command=temp_save)
    # restart
    restore_button = Button(root, text="Restart", command=restore)

    # # canvas
    # redo_button = Button(root, text="Redo", command=Redo)
    # # undo
    # undo_button = Button(root, text="Undo", command=Undo)
    # redo_button.place(x=50, y=30, width=40, height=20)
    # undo_button.place(x=100, y=30, width=40, height=20)

    # file name
    filename_label = Label(root, textvariable=filename)

    # brightness adjustment
    # brightness_up_button = Button(root, text="UP", command=lambda: img_BrightnessEdit(1))
    # brightness_down_button = Button(root, text="DOWN", command=lambda: img_BrightnessEdit(-1))

    # select file button
    # select_button = Button(root, text="Image", command=findIMG)
    # select_button.place(x=40, y=50, width=100, height=30)

    # filter
    # box_list = ttk.Combobox(root, textvariable=box_value, state="readonly")
    # box_list["values"] = ("Cartoon", "Popart", "WaterColor", "OilPainting", "PencilSketch")
    # box_list.bind("<<ComboboxSelected>>", Filter)
    # box_list.place(x=40, y=100, width=100, height=30)

    # filter button
    # cartoon_button = Button(root, text="Cartoon", command=CB)
    # cartoon_button.place(x=40, y=100, width=100, height=30)

    # waterColor_button = Button(root, text="WaterColor", command=img_watercolour)
    # waterColor_button.place(x=40, y=200, width=100, height=30)

    # oilPainting_button = Button(root, text="OilPainting", command=img_oilPainting)
    # oilPainting_button.place(x=40, y=250, width=100, height=30)

    # pencil_button = Button(root, text="PencilSketch", command=img_pencil)
    # pencil_button.place(x=40, y=300, width=100, height=30)

    # restart
    # restore_button = Button(root, text="Restart", command=restore)
    # restore_button.place(x=40, y=150, width=100, height=30)

    # save document
    # save_button = Button(root, text="Save", command=fileSave)
    # save_button.place(x=40, y=200, width=100, height=30)

    # Slice_button = Button(root, text="Slice", command=img_slice)
    # Slice_button.place(x=40, y=250, width=100, height=30)
    # Rotate slider

    # cut_button = Button(root, text="cutImage", command=openCutWin)
    # cut_button.place(x=40, y=250, width=100, height=30)
    global tip_flag
    tip_flag = 1

    # get selected file
    filename.set(filedialog.askopenfilename())
    IMG = cv2.imread(filename.get())
    IMGs = []
    position = 0
    cover = np.zeros(IMG.shape)
    Showimage(IMG, canvas, 'fit')
    conf.IMAGE_TO_TILE = "name"

    filename_label.pack(anchor="n")
    canvas.pack(padx=20, pady=20)
    root.mainloop()
