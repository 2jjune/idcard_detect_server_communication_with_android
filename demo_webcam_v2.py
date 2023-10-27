import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import tensorflow as tf
from imutils.perspective import four_point_transform
import imutils
import math
import tkinter as tk
from tkinter import Tk, Label, PhotoImage, Button
import tkinter.font as font
import tkinter.ttk as ttk
import threading
from PIL import Image, ImageTk

def delete_files(filepath):
    if os.path.exists(filepath):
        for file in os.scandir(filepath):
            os.remove(file.path)


def get_area(findCnt):
    left_up = findCnt[0]
    left_down = findCnt[1]
    right_up = findCnt[2]
    right_down = findCnt[3]

    o1 = math.atan((left_up[1] - left_down[1]) / (left_up[0] - left_down[0]))
    o2 = math.atan((right_down[1] - left_down[1]) / (right_down[0] - left_down[0]))
    angle1 = abs((o1 - o2) * 180 / math.pi)

    o3 = math.atan((left_up[1] - right_up[1]) / (left_up[0] - right_up[0]))
    o4 = math.atan((right_down[1] - right_up[1]) / (right_down[0] - right_up[0]))
    angle2 = abs((o3 - o4) * 180 / math.pi)

    area = (0.5 * math.sqrt((left_up[0] - left_down[0]) ** 2 + (left_up[1] - left_down[1]) ** 2) * math.sqrt(
        (right_down[0] - left_down[0]) ** 2 + (right_down[1] - left_down[1]) ** 2) * math.sin(math.radians(angle1))) \
           + (0.5 * math.sqrt((left_up[0] - right_up[0]) ** 2 + (left_up[1] - right_up[1]) ** 2) * math.sqrt(
        (right_down[0] - right_up[0]) ** 2 + (right_down[1] - right_up[1]) ** 2) * math.sin(math.radians(angle2)))

    return area


def make_scan_image(base_org_image, image, min_threshold=180, max_threshold=220):
    global img_size
    org_image = image.copy()
    image = cv2.resize(image, (img_size,img_size))
    ratio = org_image.shape[1] / float(image.shape[1])

    # 이미지를 grayscale로 변환하고 blur를 적용
    # 모서리를 찾기위한 이미지 연산
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, min_threshold, max_threshold)

    # contours를 찾아 크기순으로 정렬
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    findCnt = None
    # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 영역으로 판단하고 break
        if len(approx) == 4:
            findCnt = approx
            break
    # 만약 추출한 윤곽이 없을 경우 오류
    if findCnt is None:
        return 0
        # raise Exception(("Could not find outline."))
    area = get_area(findCnt.reshape(4,2))
    print(area)

    if area > (image.shape[0]**2)*0.82 or area < (image.shape[0]**2)*0.1 or np.isnan(area) == True:
        return 0
    # 원본 이미지에 찾은 윤곽을 기준으로 이미지를 보정
    findCnt = findCnt.reshape(4,2)
    for i in range(4):
        findCnt[i][0] *= (base_org_image.shape[1]/img_size)
        findCnt[i][1] *= (base_org_image.shape[0]/img_size)
    print(findCnt)
    transform_image = four_point_transform(base_org_image, findCnt)
    transform_image = cv2.resize(transform_image, (img_size, img_size))

    return transform_image

# def close_app():
#     global pic_window
#     pic_window.destroy()

def close_cam_window_app():
    global cam_window
    cam_window.destroy()

def close_app():
   global win
   win.destroy()

def close_toplevel_app():
    global new_window
    new_window.destroy()

def check_5pic():
    global win, new_window
    if len(os.listdir('C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame')) < 5:
        new_window = tk.Toplevel(win)
        # pic_window.update()
        # new_win = tk.Toplevel(cam_window)
        label = tk.Label(new_window, text='사진을 더 찍으세요', font=font.Font(size=20, weight='bold'), fg='red')
        label.pack(side='top')
        label.pack(pady=5)
        label = tk.Label(new_window, text='워핑된 사진 : {}'.format(
            len(os.listdir('C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame'))), font=35)
        label.pack(side='top')
        label.pack(pady=5)
        label = tk.Button(new_window, text='확인', font=font.Font(size=15, weight='bold', underline=1), command=close_toplevel_app)
        label.pack(side='bottom')
        label.pack(pady=5)
        # pic_window.mainloop()

        print('사진을 더 찍으세요')
    else:
        win.destroy()

def screen_shot():
   global num, webcam_img_list, webcam_Path, warp_model, img_size
   # global cv2image
   # cv2image= cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
   cv2image= cap.read()[1]
   cv2image = cv2image[:, int(cv2image.shape[1] * 0.125):int(cv2image.shape[1] * 0.875)]
   # cv2.imwrite(f"C:/Users/USER/PycharmProjects/homography/segment_result/webcam/capture_{num}.jpg", cv2image)


   #워핑 하는 부분----------------------------------------------------------------

   image = cv2.resize(cv2image, (img_size, img_size))
   image = np.asarray(image)
   image = image / 255.
   image = np.expand_dims(image, axis=0)

   pred_mask = warp_model.predict(image)

   pred_mask = np.squeeze(pred_mask, axis=0)
   pred_mask2 = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 1))

   for i in range(pred_mask.shape[0]):
       for j in range(pred_mask.shape[1]):
           if pred_mask[i][j] > 0.5:
               pred_mask2[i][j] = 255
   save_img = tf.keras.utils.array_to_img(pred_mask2)
   # save_img.save('./segment_result/{0}_{1}.jpg'.format(file[:-4],check_num))

   save_img = np.array(save_img)
   save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)

   receipt_image = make_scan_image(cv2image, save_img, min_threshold=25, max_threshold=70)

   if type(receipt_image) == int:
       pass
   else:
       cv2.imwrite(f'C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame/capture_{num}.jpg',receipt_image)

   num += 1


# Define function to show frame
def show_frames():
   # Get the latest frame and convert into Image
   global  cv2image, cap, label
   # num_of_file = "pic:" + str(len(os.listdir('C:/Users/USER/PycharmProjects/homography/segment_result/webcam/')))
   num_of_warp_file = "warp:" + str(len(os.listdir('C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame/')))

   # cap_img = cv2.putText(cap.read()[1], num_of_file, (1550, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness=8)
   # cap_img = cv2.putText(cap_img, num_of_warp_file, (1000, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness=8)
   cap_img = cv2.putText(cap.read()[1], num_of_warp_file, (1550, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness=8)
   cv2image= cv2.cvtColor(cap_img,cv2.COLOR_BGR2RGB)
   # cv2.imshow('aaa',cv2image)

   small = cv2.resize(cv2image, (1024, 576))
   img = Image.fromarray(small)
   # Convert image to PhotoImage
   imgtk = ImageTk.PhotoImage(image = img)
   label.imgtk = imgtk
   label.configure(image=imgtk)
   # Repeat after an interval to capture continiously
   label.after(20, show_frames)


class check_finish:
   q = 0
   def __init__(self):
      global win
      btn = Button(win, text="종료", command=self.check_finish_yes, width=7, height=2)
      # btn.pack(padx=12)
      # btn.place(x=600, y=450)
      btn.pack(side='bottom')
      btn.pack(pady=40)

      # width = win.winfo_screenwidth()
      # height = win.winfo_screenheight()
      # print(width, height)

      # win.mainloop()

   def __repr__(self):
      return "% s" % (self.q)

   def check_finish_yes(self):
      self.q = 1
      win.destroy()
      return self.q


def webcam(a):
   # Create an instance of TKinter Window or frame
   q=0
   global cap, label, cv2image, win, num
   num=a
   # print(a)
   win = Tk()
   # Set the size of the window
   win.geometry("1280x720")
   win.title("NEW WINDOW")

   # Create a Label to capture the Video frames
   # label=Label(win, text= "Tkinter is a GUI Library in Python", font=('Helvetica 15 bold')).pack(pady=20)
   label = Label(win)
   # label.grid(row=1, column=1)
   label.pack(side='top')
   cap = cv2.VideoCapture(1)
   cv2image = None
   # new_win= Toplevel(win)
   # new_win.geometry("1024x576")
   button = tk.Button(win, text="완료", command=check_5pic, width=15, height=2, bg='white', fg='red',
                      overrelief='solid', font=font.Font(size=13, weight='bold', underline=1))
   button2 = tk.Button(win, text="촬영", command=screen_shot, width=15, height=2, bg='white', overrelief='solid', font=font.Font(size=13, weight='bold', underline=1))
   # button.place(x=382, y=540)
   # button2.place(x=552, y=540)
   # button.grid(row = 3, column=1)
   # button2.grid(row = 3, column=2)
   button.pack(side='left')
   button2.pack(side='right')
   button.pack(padx=200)
   button2.pack(padx=200)

   q = check_finish()

   show_frames()
   win.mainloop()

   return (num,q)


def main(model):
    a = 0
    global webcam_Path, webcam_img_list, warp_model, img_size
    img_size = 512
    warp_model = model
    while(1):
        webcam_Path = 'C:/Users/USER/PycharmProjects/homography/segment_result/webcam/'
        delete_files(webcam_Path)
        a,q = webcam(a)


        webcam_img_list = os.listdir(webcam_Path)
        frames = []
        if repr(q)=="1":
            return 'q'

        # global cam_window
        # cam_window = Tk()
        # label = tk.Label(cam_window, text='사진을 전처리 중입니다..', font=font.Font(size=15, weight='bold'))
        # label.pack(side='top')
        # label.pack(pady=5)
        #
        # progressbar_indeter = ttk.Progressbar(cam_window, maximum=30, mode='indeterminate')
        # progressbar_indeter.start(10)
        # progressbar_indeter.pack()
        #
        # thread = threading.Thread(target=warp, args=(model,progressbar_indeter))
        # thread.start()
        #
        # cam_window.mainloop()

        if len(os.listdir('C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame'))<5:

            new_window = tk.Toplevel(win)
            # pic_window.update()

            # new_win = tk.Toplevel(cam_window)
            label = tk.Label(new_window, text='사진을 더 찍으세요', font=font.Font(size=20, weight='bold'), fg='red')
            label.pack(side='top')
            label.pack(pady=5)
            label = tk.Label(new_window, text='워핑된 사진 : {}'.format(len(os.listdir('C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame'))), font=35)
            label.pack(side='top')
            label.pack(pady=5)
            label = tk.Button(new_window, text='확인', font=font.Font(size=15, weight='bold', underline=1), command=close_app)
            label.pack(side='bottom')
            label.pack(pady=5)
            # pic_window.mainloop()

            print('사진을 더 찍으세요')

        else:
            break

if __name__ == "__main__":
	main()