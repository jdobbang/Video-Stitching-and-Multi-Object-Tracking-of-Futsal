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

Warping using SIFT algorithm
![img](./img/frame.png)
Warping by user
![img](./img/HomographyControl.png)
Warping by user - crop 
![img](./img/frame00001.jpg)

## Tracking Frame 
We use DeepSORT algorithm to track multi-objects(players,ball)

Track the initial version of stitching
![img](./img/tracking.JPG)
Track the second version of stitching + coordinate point
![img](./img/tracking.JPG)  

## data output(hitmap,average location)

We plot some data using .csv dtat including objects coordinate

hit map
![img](./img/hitmap.JPG)

average location
![img](./img/average_location.JPG)

## To do list  
- [ ] perspective warping of coordinates
- [ ] add ball class
- [ ] raise the accuracy of tracking


