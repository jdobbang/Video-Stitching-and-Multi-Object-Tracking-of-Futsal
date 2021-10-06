#마스크 단순
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def feature_matching(img1, img2, savefig=False):
    
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches2to1 = flann.knnMatch(des2,des1,k=2)

    matchesMask_ratio = [[0,0] for i in range(len(matches2to1))]
    match_dict = {}
    for i,(m,n) in enumerate(matches2to1):
        if m.distance < 0.7*n.distance:
            matchesMask_ratio[i]=[1,0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1,des2,k=2)
    matchesMask_ratio_recip = [[0,0] for i in range(len(recip_matches))]

    for i,(m,n) in enumerate(recip_matches):
        if m.distance < 0.7*n.distance: # ratio
            if m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx: #reciprocal
                good.append(m)
                matchesMask_ratio_recip[i]=[1,0]

    if savefig:
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask_ratio_recip,
                           flags = 0)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,recip_matches,None,**draw_params)

        plt.figure(),plt.xticks([]),plt.yticks([])
        plt.imshow(img3,)
        plt.savefig("feature_matching.png",bbox_inches='tight')

    return ([ kp1[m.queryIdx].pt for m in good ],[ kp2[m.trainIdx].pt for m in good ])

def cylindricalWarpImage(img1, K, savefig=False):
    f = K[0,0]
    if len(img1.shape) == 3:
        im_h,im_w,im_c = img1.shape
        
        # go inverse from cylindrical coord to the image
        # (this way there are no gaps)
        cyl = np.zeros_like(img1)
        cyl_mask = np.zeros_like(img1)
        cyl_h,cyl_w,cyl_c = cyl.shape
        x_c = float(cyl_w) / 2.0
        y_c = float(cyl_h) / 2.0
        for x_cyl in np.arange(0,cyl_w):
            for y_cyl in np.arange(0,cyl_h):
                theta = (x_cyl - x_c) / f
                h     = (y_cyl - y_c) / f
    
                X = np.array([math.sin(theta), h, math.cos(theta)])
                X = np.dot(K,X)
                x_im = X[0] / X[2]
                if x_im < 0 or x_im >= im_w:
                    continue
    
                y_im = X[1] / X[2]
                if y_im < 0 or y_im >= im_h:
                    continue
    
                cyl[int(y_cyl),int(x_cyl)] = img1[int(y_im),int(x_im)]
                cyl_mask[int(y_cyl),int(x_cyl)] = 255

    if len(img1.shape) == 2:
        im_h,im_w = img1.shape
    
        # go inverse from cylindrical coord to the image
        # (this way there are no gaps)
        cyl = np.zeros_like(img1)
        cyl_mask = np.zeros_like(img1)
        cyl_h,cyl_w = cyl.shape
        x_c = float(cyl_w) / 2.0
        y_c = float(cyl_h) / 2.0
        for x_cyl in np.arange(0,cyl_w):
            for y_cyl in np.arange(0,cyl_h):
                theta = (x_cyl - x_c) / f
                h     = (y_cyl - y_c) / f
    
                X = np.array([math.sin(theta), h, math.cos(theta)])
                X = np.dot(K,X)
                x_im = X[0] / X[2]
                if x_im < 0 or x_im >= im_w:
                    continue
    
                y_im = X[1] / X[2]
                if y_im < 0 or y_im >= im_h:
                    continue
    
                cyl[int(y_cyl),int(x_cyl)] = img1[int(y_im),int(x_im)]
                cyl_mask[int(y_cyl),int(x_cyl)] = 255


    return (cyl,cyl_mask)

def getTransform(src, dst, method='affine'):
    pts1,pts2 = feature_matching(src,dst)

    src_pts = np.float32(pts1).reshape(-1,1,2)
    dst_pts = np.float32(pts2).reshape(-1,1,2)

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)

    if method == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    return (M, pts1, pts2, mask)
   
def Laplacian_blending(img1, img2):
    
    levels = 3
    # generating Gaussian pyramids for both images
    gpImg1 = [img1.astype('float32')]
    gpImg2 = [img2.astype('float32')]
    for i in range(levels):
        img1 = cv2.pyrDown(img1)   # Downsampling using Gaussian filter
        gpImg1.append(img1.astype('float32'))
        img2 = cv2.pyrDown(img2)
        gpImg2.append(img2.astype('float32'))

    # Generating Laplacin pyramids for both images
    lpImg1 = [gpImg1[levels]]
    lpImg2 = [gpImg2[levels]]

    for i in range(levels,0,-1):
        # Upsampling and subtracting from upper level Gaussian pyramid image to get Laplacin pyramid image
        tmp = cv2.pyrUp(gpImg1[i]).astype('float32')
        tmp = cv2.resize(tmp, (gpImg1[i-1].shape[1],gpImg1[i-1].shape[0]))
        lpImg1.append(np.subtract(gpImg1[i-1],tmp))

        tmp = cv2.pyrUp(gpImg2[i]).astype('float32')
        tmp = cv2.resize(tmp, (gpImg2[i-1].shape[1],gpImg2[i-1].shape[0]))
        lpImg2.append(np.subtract(gpImg2[i-1],tmp))

    laplacianList = []
    for lImg1,lImg2 in zip(lpImg1,lpImg2):
        rows,cols,c = lImg1.shape
        # Merging first and second half of first and second images respectively at each level in pyramid
        mask1 = np.zeros(lImg1.shape)
        mask2 = np.zeros(lImg2.shape)
        mask1[:, 0:int(cols * 0.67 )] = 1 #test image : 1334 x 750 
        mask2[:, int(cols *0.67 ):] = 1
        
        tmp1 = np.multiply(lImg1, mask1.astype('float32'))
        tmp2 = np.multiply(lImg2, mask2.astype('float32'))
        tmp = tmp1 + tmp2
        laplacianList.append(tmp)
    
    img_out = laplacianList[0]
    for i in range(1,levels+1):
        img_out = cv2.pyrUp(img_out)   # Upsampling the image and merging with higher resolution level image
        img_out = cv2.resize(img_out, (laplacianList[i].shape[1],laplacianList[i].shape[0]))
        img_out = np.add(img_out, laplacianList[i])
    
    np.clip(img_out, 0, 255, out=img_out)
    return img_out.astype('uint8')

def Laplacian_cylindrical_warping(M21,img1, img2):
    
    # Write your codes here
    h,w,c = img1.shape
    f =520
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    (img1cyl, mask1) = cylindricalWarpImage(img1, K)  #Returns: (image, mask)
    (img2cyl, mask2) = cylindricalWarpImage(img2, K)
    #(img3cyl, mask3) = cylindricalWarpImage(img3, K)
    
    # Add padding to allow space around the center image to paste other images.
    img1cyl = cv2.copyMakeBorder(img1cyl,50,50,300,300, cv2.BORDER_CONSTANT)    
    transformedImage2 = cv2.warpAffine(img2cyl, M21, (img1cyl.shape[1],img1cyl.shape[0]))

    # Stich image1 and image2 using mask.
    transformedImage21 = Laplacian_blending( img1cyl,transformedImage2)

    return transformedImage21

def Laplacian_cylindrical_warping_Calculation(img1, img2):
    
    # Write your codes here
    h,w = img1.shape
    f =520
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    (img1cyl, mask1) = cylindricalWarpImage(img1, K)  #Returns: (image, mask)
    (img2cyl, mask2) = cylindricalWarpImage(img2, K)

    # Add padding to allow space around the center image to paste other images.
    img1cyl = cv2.copyMakeBorder(img1cyl,50,50,300,300, cv2.BORDER_CONSTANT)
    
    # Calculate Affine transformation to transform image2 in plane of image1.
    (M21, pts2, pts1, mask4) = getTransform(img2cyl, img1cyl, 'affine')

    return M21


def stitch(ltx,lty,rbx,rby,left_video,right_video):

    #cap1 = cv2.VideoCapture("./input/video_left.mp4")
    #cap2 = cv2.VideoCapture("./input/video_right.mp4")

    cap1 = cv2.VideoCapture(left_video)
    cap2 = cv2.VideoCapture(right_video)
    
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    #gray scale
    input_image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    input_image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    #resize gray
    image1_resize = cv2.resize(input_image1, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    image2_resize = cv2.resize(input_image2, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    #calculate matrix
    M21 = Laplacian_cylindrical_warping_Calculation(image1_resize,image2_resize)
    i=1
    print("Stithing.....making frames.....")
    while(1):
        #read frame
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read() 
        
        #gray scale : 일단은 option
        #input_image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        #input_image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        input_image1 = frame1
        input_image2 = frame2

        #resize
        image1_resize = cv2.resize(input_image1, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
        image2_resize = cv2.resize(input_image2, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
        
        # Call the function and show the frames
        output1 = Laplacian_cylindrical_warping(M21,image1_resize, image2_resize)
        output2 = output1[int(lty):int(rby),int(ltx):int(rbx)]
        
        filename = str('./cylindricalFrames/') + str(i).zfill(5) + str('.jpg')
        cv2.imwrite(filename,output2)
        
        print(i,"번째 frame")
        i+=1
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    print("Stithing Complete")
    