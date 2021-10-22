# Futsal-Image-Playing-Analysis-Service 

work added : webRTC(by chaeyeon4u), kinesis(by hkjs96) , MOTR(by heosuab)

## You can watch demo video from here : https://www.youtube.com/watch?v=t8mXEbXQEL0

## Goal 
Stitching multi-videos into an entire view of futsal game and analyze it by tracking players and a ball

## Creating Top down view Video 
We expect that we can take the top-down view by this 4m tripod at all type of futsal field   
<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/6c883fdc7a7acb1867168b5a27a8355951b6201e/img/camera.jpg" width = "200">

# WebRTC(Real-Time-Communication) and Video Record

---

- Web RTC(Real-Time-Communication) with Kotlin
- P2P Network
- Create Key
- Video Record
- Login / Signup

## 실시간 모니터링

< Create Room Key >

사용자가 입력한 키가 유일한 키이면 방 생성.

키가 다른 사람이 만든 키와 중복될 경우 방 생성 불가.

<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/a434084d9ecc4355b1092a2a30bc5e0783b084a4/img/ezgif.com-gif-maker.gif" width = "200">


< WebRTC Start & Record Start >

<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/a434084d9ecc4355b1092a2a30bc5e0783b084a4/img/ezgif.com-gif-maker2.gif" width = "200">

< 카메라 전환과 RTC & Record >

<img src = "https://user-images.githubusercontent.com/55488114/129470670-a70d71e7-9dec-4bbd-8445-4026ff578390.gif" width = "200">

<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/a434084d9ecc4355b1092a2a30bc5e0783b084a4/img/ezgif.com-gif-maker3.gif" width = "200">

## 영상 녹화 결과 영상 (일부)

<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/a434084d9ecc4355b1092a2a30bc5e0783b084a4/img/ezgif.com-gif-maker4.gif" width = "200">
     

# AWS Kinesis Streaming

Monitoring scenes from two device on a single page
     
<img src ="https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/a434084d9ecc4355b1092a2a30bc5e0783b084a4/img/kinesis.jpg" width = "200">

## 로그인

<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/a434084d9ecc4355b1092a2a30bc5e0783b084a4/img/Login.png" width = "200" >

## 회원가입

<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/a434084d9ecc4355b1092a2a30bc5e0783b084a4/img/Signup.png" width = "200" >
     
## 메뉴
     
<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/a434084d9ecc4355b1092a2a30bc5e0783b084a4/img/menu.png" width="200">
    
### Stitching and Tracking two videos by Tkinter GUI Program (step1~step )
<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/646b6fbc91d56f845c8e6567a7147af7f8c62c8f/img/guiSample.png" width = "200">

## Step1: Open Left, Right Source Frame 
<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/646b6fbc91d56f845c8e6567a7147af7f8c62c8f/img/left.JPG" width = "200">
<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/646b6fbc91d56f845c8e6567a7147af7f8c62c8f/img/right.JPG" width = "200">

## Step2 : Check a Pre-Stitched Frame and cut a video
We need two decide 6 points. 2 points for cropping and 4 points for converting coordinates to the real data.

## Step3 : Stitching
Version 1.0 : Perspective transform 

<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/646b6fbc91d56f845c8e6567a7147af7f8c62c8f/img/frame.png" width = "200">

Version 2.0 : User defined homography 

<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/646b6fbc91d56f845c8e6567a7147af7f8c62c8f/img/HomographyControl.png" width = "200">

Version 3.0 : Warping by Cylindrical Laplcian Blending

<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/0380f7335d544f9680e37bb49efe4f53a08c0c8a/img/cylindricalFrame.png" width = "200">

## Step4: Tracking
We use DeepSORT algorithm to track multi-objects(players,ball) and produce csv coordinate file

Version 1.0 : DeepSORT + YOLOv3

<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/c7fc592a3e6bb6019d8cf6903cb1dd4a6ac015c8/img/tracking+.jpg" width ="200">
     
Version 2.0 (DeepSORT + YOLOv4

<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/00021117bde96fe45e082746e3573b34c4347de0/img/YOLOv4_tracking.png" width="200" >
     
Version 2.1 : DeepSORT + YOLO4

<img src = " https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/0380f7335d544f9680e37bb49efe4f53a08c0c8a/img/cylindrical_tracking_frame.jpg " width = "200" >

Version 3.0 : MOTR(Improved ID switching issue)

<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/a434084d9ecc4355b1092a2a30bc5e0783b084a4/img/kinesis.jpg" width = "200" >
     
## Data Visualization

We visualzied and calculate heatmap, minimap(video), average position map, activity, spped. We described those on the Flask webServer 

<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/0380f7335d544f9680e37bb49efe4f53a08c0c8a/img/webSample.png" width = "200">


