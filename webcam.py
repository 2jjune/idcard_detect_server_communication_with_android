import cv2
import tkinter as tk # Tkinter
from PIL import ImageTk, Image # Pillow
import cv2 as cv # OpenCV
import os
import threading
from tkinter import *
from PIL import Image, ImageTk
import cv2
from tkinter import ttk
import tkinter.font as font

def close_app():
   global win
   win.destroy()

def screen_shot():
   global num
   # global cv2image
   # cv2image= cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
   cv2image= cap.read()[1]
   cv2image = cv2image[:, int(cv2image.shape[1] * 0.125):int(cv2image.shape[1] * 0.875)]
   cv2.imwrite(f"C:/Users/USER/PycharmProjects/homography/segment_result/webcam/capture_{num}.jpg", cv2image)
   num += 1

# Define function to show frame
def show_frames():
   # Get the latest frame and convert into Image
   global  cv2image, cap, label
   num_of_file = "pic:" + str(len(os.listdir('C:/Users/USER/PycharmProjects/homography/segment_result/webcam/')))
   num_of_warp_file = "warp:" + str(len(os.listdir('C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame/')))

   cap_img = cv2.putText(cap.read()[1], num_of_file, (1550, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness=8)
   cap_img = cv2.putText(cap_img, num_of_warp_file, (1000, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness=8)
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

def main(a):
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
   cap = cv2.VideoCapture(0)
   cv2image = None
   # new_win= Toplevel(win)
   # new_win.geometry("1024x576")
   button = tk.Button(win, text="완료", command=close_app, width=15, height=2, bg='white', fg='red',
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


if __name__ == "__main__":
   main()

# main()

# def main():
#     a=0
#     # global lbl
#     webcam = cv2.VideoCapture(1)
#     width = webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
#     height = webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)
#
#     # win = tk.Tk()  # 인스턴스 생성
#     # win.title("AniWatch")  # 제목 표시줄 추가
#     # win.geometry("{0}x{1}+50+50".format(round(width/2+100),round(height/2+100)))  # 지오메트리: 너비x높이+x좌표+y좌표
#     # win.resizable(False, False)  # x축, y축 크기 조정 비활성화
#     #
#     # lbl = tk.Label(win, text="Tkinter와 OpenCV를 이용한 GUI 프로그래밍")
#     # lbl.grid(row=0, column=0)  # 라벨 행, 열 배치
#     #
#     # frm = tk.Frame(win, bg="white", width=round(width/2), height=round(height/2))  # 프레임 너비, 높이 설정
#     # frm.grid(row=1, column=0)  # 격자 행, 열 배치
#     # lbl1 = tk.Label(frm)
#     # lbl1.grid()
#     #
#     # win.mainloop()  # GUI 시작
#
#     if not webcam.isOpened():
#         print("Could not open webcam")
#         exit()
#
#     while webcam.isOpened():
#         status, frame = webcam.read()
#
#         if status:
#             # print((round(width/2),round(height/2)))
#             small = cv2.resize(frame,(1024,576))#1280,720
#             cv2.imshow("test", small)
#             # if cv2.waitKey(1) != -1:
#             if cv2.waitKey(1) & 0xFF == ord('s'):
#                 frame = frame[:, int(frame.shape[1] * 0.125):int(frame.shape[1] * 0.875)]
#                 cv2.imwrite('C:/Users/USER/PycharmProjects/homography/segment_result/webcam/capture_{}.jpg'.format(a), frame)
#                 a += 1
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     webcam.release()
#     cv2.destroyAllWindows()
#     return a
#
# if __name__ == "__main__":
# 	main()