# Futsal-Image-Playing-Analysis-Service
## Goal 
Stitching multi-videos into an entire view of futsal game and analyze it by tracking players and a ball

## Camera : Top down view ( 4m ) 
We expect that we can take the top-down view by this tripod at all type of futsal field   
![img](./img/camera.jpg)

## Left, Right Source Frame : filmed by smart phones 
![img](./img/left.JPG)
![img](./img/right.JPG) 

## Stitched Frame 
We need stitching of two videos for the entire view of the field

Warping using SIFT algorithm(@sift_video.cpp)
![img](./img/frame.png)
Warping by user defined homography 
![img](./img/HomographyControl.png)
Crop version( user defined homography)(@ StitchingControl.cpp)
![img](./img/frame00001.jpg)

## Tracking Frame 
We use DeepSORT algorithm to track multi-objects(players,ball)

Track the initial version of stitching (DeepSORT + YOLOv3)
![img](./img/tracking.JPG)
Track the user defined homography stitching + coordinate point (DeepSORT + YOLOv3 , @object_trackin.py)
![img](./img/tracking+.jpg)
Track the user defined homography stitching + cooordinate point + ball class added (DeepSORT + YOLOv4)
![img](./img/YOLOv4_tracking.png)
## Data Visualization

We first have to convert pixel coordinate to real world coordination by calculating perspective homography

field width : 17.06 m / field heigth : 7.05 m (multiply 28 for easy visualization)

pixel coordinate : left-top [165,130] , left-bottom [2,270] , right-bottom [870,263], right-top [701,131]

hit map of person_ID_1

![img](./img/heatMap.JPG)

average location of person_ID_1

![img](./img/averageLocation.JPG)



