# Futsal-Image-Playing-Analysis-Service

## You can watch demo video from here : https://www.youtube.com/watch?v=t8mXEbXQEL0

## Goal 
Stitching multi-videos into an entire view of futsal game and analyze it by tracking players and a ball

## Creating Top down view Video 
We expect that we can take the top-down view by this 4m tripod at all type of futsal field   
<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/6c883fdc7a7acb1867168b5a27a8355951b6201e/img/camera.jpg" width = "200">

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

<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/0380f7335d544f9680e37bb49efe4f53a08c0c8a/img/cylindricalFrame.png" width = "200"

## Step4: Tracking
We use DeepSORT algorithm to track multi-objects(players,ball) and produce csv coordinate file

Version 1.0 : DeepSORT + YOLOv3
![img](./img/tracking.JPG)
![img](./img/tracking+.jpg)

Version 2.0 (DeepSORT + YOLOv4

<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/0380f7335d544f9680e37bb49efe4f53a08c0c8a/img/tracking+.jpg" width = "200">
     
Version 2.1 : DeepSORT + YOLO4

<img src = " https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/0380f7335d544f9680e37bb49efe4f53a08c0c8a/img/cylindrical_tracking_frame.jpg " width = "200" >

## Data Visualization

We visualzied and calculate heatmap, minimap(video), average position map, activity, spped. We described those on the Flask webServer 

<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/0380f7335d544f9680e37bb49efe4f53a08c0c8a/img/webSample.png" width = "200">


