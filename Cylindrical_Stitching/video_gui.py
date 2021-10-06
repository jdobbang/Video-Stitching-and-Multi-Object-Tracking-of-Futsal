from tkinter import*
from tkinter.filedialog import*
import glob
import cv2
import numpy as np
import time

#function files
import stitching as st
import video_trackbar as vt
import frames2Video as f2v
import edit
import stitching_coordinate as stc

#임시 function
def donothing():
   filewin = Toplevel(window)
   button = Button(filewin, text="Do nothing button")
   button.pack()

#왼쪽 비디오 선택
def OpenFileL(file_record ):
    file_record['left_video'] =  askopenfilename(title = "Select Left Video",filetypes = (("video files1","*.mp4"),("video files2","*.avi")))
    print("왼쪽 비디오 선택")
    
#오른쪽 비디오 선택
def OpenFileR(file_record ):
    file_record['right_video'] =  askopenfilename(title = "Select Right Video",filetypes = (("video files1","*.mp4"),("video files2","*.avi")))
    print("오른쪽 비디오 선택")
    
#window 선언
window = Tk()
filename_record = {}

# 제목, 사이즈, 크기 조절
window.title("Supput")

window.geometry("600x500")

#메뉴
menubar = Menu(window)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Left_Video",command = lambda: OpenFileL(filename_record))
filemenu.add_command(label="Right_Video",command = lambda: OpenFileR(filename_record))
menubar.add_cascade(label="File",menu=filemenu)

videomenu = Menu(menubar, tearoff=0)
videomenu.add_command(label="Edit",command = lambda : vt.trackbar(filename_record['left_video'],filename_record['right_video']))
videomenu.add_command(label="Stitchg Confirm",command = lambda : stc.stitch())
videomenu.add_command(label="make video",command = lambda : f2v.makeVideo())
videomenu.add_command(label="Track",command = lambda : donothing)
menubar.add_cascade(label="Video",menu=videomenu)
window.config(menu=menubar)


#텍스트 입력(프레임 간격)

txt1 = Text(window,width=30,height=5)
txt1.pack()
txt1.insert(END,"좌측 비디오의 시작,종료 프레임 적으세요")
e1 = Entry(window, width = 30)
e1.pack()
e1.insert(0,"시작 프레임")

e2 = Entry(window, width = 30)
e2.pack()
e2.insert(0,"종료 프레임")

btn2 =  Button(window, text = "자르기", command = lambda: edit.cut('./input/video_left.mp4', e1.get(), e2.get()))
btn2.pack()

txt2 = Text(window,width=30,height=5)
txt2.pack()
txt2.insert(END,"우측 비디오의 시작,종료 프레임 적으세요")
e3 = Entry(window, width = 30)
e3.pack()
e3.insert(0,"시작 프레임")

e4 = Entry(window, width = 30)
e4.pack()
e4.insert(0,"종료 프레임")

btn4 =  Button(window, text = "자르기", command = lambda: edit.cut('./input/video_right.mp4',e3.get(),e4.get()))
btn4.pack()

# 네 모서리 입력

txt3 = Text(window,width=30,height=5)
txt3.pack()
txt3.insert(END,"좌상x, 좌상y, 우하x, 우하y 순서로 네 모서리의 좌표를 입력하세요")
e5 = Entry(window, width = 30)
e5.pack()
e5.insert(0,"좌상x")

e6 = Entry(window, width = 30)
e6.pack()
e6.insert(0,"좌상y")

e7 = Entry(window, width = 30)
e7.pack()
e7.insert(0,"우상x")

e8 = Entry(window, width = 30)
e8.pack()
e8.insert(0,"우상y")

btn3 =  Button(window, text = "프레임 생성", command = lambda: st.stitch(e5.get(),e6.get(),e7.get(),e8.get(),filename_record['left_video'],filename_record['right_video']) )
btn3.pack()

btn4 = Button(window, text = "동영상 생성", command = lambda:f2v.makeVideo())
btn4.pack()

window.mainloop()