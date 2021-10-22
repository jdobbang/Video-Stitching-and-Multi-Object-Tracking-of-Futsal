# Futsal-Image-Playing-Analysis-Service

## You can watch demo video from here : https://www.youtube.com/watch?v=t8mXEbXQEL0

## Goal 
Stitching multi-videos into an entire view of futsal game and analyze it by tracking players and a ball

## Creating Top down view Video 
We expect that we can take the top-down view by this 4m tripod at all type of futsal field   
<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/6c883fdc7a7acb1867168b5a27a8355951b6201e/img/camera.jpg" width = "200">

### Stitching and Tracking two videos by Tkinter GUI Program
<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/646b6fbc91d56f845c8e6567a7147af7f8c62c8f/img/guiSample.png" width = "200">

## Open Left, Right Source Frame 
<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/646b6fbc91d56f845c8e6567a7147af7f8c62c8f/img/left.JPG" width = "200">
<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/646b6fbc91d56f845c8e6567a7147af7f8c62c8f/img/right.JPG" width = "200">

## Check a Pre-Stitched Frame 
We need two decide 6 points. 2 points for cropping and 4 points for converting coordinates to the real data.

![img](./img/mouseClick.png)

## Stitching
Version 1.0 : Perspective transform 
<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/646b6fbc91d56f845c8e6567a7147af7f8c62c8f/img/frame.png" width = "200">

Version 2.0 : User defined homography 
<img src = "https://github.com/jdobbang/Video-Stitching-and-Multi-Object-Tracking-of-Futsal/blob/646b6fbc91d56f845c8e6567a7147af7f8c62c8f/img/HomographyControl.png" width = "200">

Version 3.0 : Warping by Cylindrical Laplcian Blending
![img](./Cylindrical_Stitching/cylindricalFrames_new/00001.jpg)

## Tracking
We use DeepSORT algorithm to track multi-objects(players,ball)

Version 1.0 : DeepSORT + YOLOv3
![img](./img/tracking.JPG)
![img](./img/tracking+.jpg)

Version 2.0 (DeepSORT + YOLOv4
![img](./img/YOLOv4_tracking.png)

Version 2.1 : DeepSORT + YOLO4
![img](./img/cylindrical_tracking_frame.jpg)

## Data Visualization

We first have to convert pixel coordinate to real world coordination by calculating perspective homography

field width : 17.06 m / field heigth : 7.05 m (multiply 28 for easy visualization)

pixel coordinate : left-top [165,130] , left-bottom [2,270] , right-bottom [870,263], right-top [701,131]

Numerical Data1 : Activity + Speed
![img](./img/activity_speed.png)

Visual Data1 : hit map of person_ID_1

![img](./img/heatMap.JPG)

Visual Data2 : average location of person_ID_1

![img](./img/averageLocation.JPG)

Visual Data3 : minimap

original minimap

![img](./img/minimap_original.png)

line added minimap

![img](./img/minimap_line.png)

Flask webServer 

![img](./img/webSample.png)


