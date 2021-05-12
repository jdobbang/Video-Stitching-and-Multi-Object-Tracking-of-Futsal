# Instructions:
# Do not change the output file names, use the helper functions as you see fit
import os
import sys
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

    matchesMask_ratio = [[0,0] for i in range(len(matches2to1))] # xrange 에서 range로 수정
    match_dict = {}
    for i,(m,n) in enumerate(matches2to1):
        if m.distance < 0.7*n.distance:
            matchesMask_ratio[i]=[1,0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1,des2,k=2)
    matchesMask_ratio_recip = [[0,0] for i in range(len(recip_matches))]# xrange 에서 range로 수정

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

#원기둥 좌표계 상 warping
def cylindricalWarpImage(img1, K, savefig=False):
    f = K[0,0]

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


    if savefig:
        plt.imshow(cyl, cmap='gray')
        plt.savefig("cyl.png",bbox_inches='tight')

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
   
#라플라시안 블렌딩
def Laplacian_blending(img1, img2, mask1,mask2):
    
    levels = 5
    # generating 가우시안 피라미드 생성
    gpImg1list = [img1]
    gpImg2list = [img2]
    
    for i in range(levels):
        img1 = cv2.pyrDown(img1)   # Downsampling using Gaussian filter,image1
        gpImg1list.append(img1)

        img2 = cv2.pyrDown(img2) # Downsampling using Gaussian filter,image1
        gpImg2list.append(img2)

    gaussiansumList = []
    #가우시안 피라미드를 활용하여 합 피라미드 생
    for gImg1,gImg2 in zip(gpImg1list,gpImg2list):
        #cv2.imshow("gImg1",gImg1)
        #cv2.imshow("gImg2",gImg2)
        rows,cols = gImg1.shape
        
        mask1_and=cv2.bitwise_and(mask1,mask2)
        mask1_xor=cv2.bitwise_xor(mask1,mask1_and)
        mask1_xor = cv2.resize(mask1_xor,(cols,rows),interpolation = cv2.INTER_CUBIC)
        mask2_ = cv2.resize(mask2,(cols,rows),interpolation = cv2.INTER_CUBIC)

        tmp1 = cv2.bitwise_and(gImg1, mask1_xor)
        tmp2 = cv2.bitwise_and(gImg2, mask2_)
        tmp = tmp1 + tmp2
        #cv2.imshow("tmp",tmp)
        #cv2.waitKey(0)
        gaussiansumList.append(tmp) # 합 피라미

    # Generating Laplacin pyramids for both images, 피라미드 마지막 이미지
    lpImg1 = [gpImg1list[levels]]
    lpImg2 = [gpImg2list[levels]]
    
    #laplacian pyramid for each image
    for i in range(levels,0,-1):
        # Upsampling and subtracting from upper level Gaussian pyramid image to get Laplacin pyramid image
        #image1
        tmp = cv2.pyrUp(gpImg1list[i])
        tmp = cv2.resize(tmp, (gpImg1list[i-1].shape[1],gpImg1list[i-1].shape[0]))#사이즈 조절
        lpImg1.append(np.subtract(gpImg1list[i-1],tmp))
        cv2.imshow("img1_laplacian",tmp)    
        #image2
        tmp = cv2.pyrUp(gpImg2list[i])
        tmp = cv2.resize(tmp, (gpImg2list[i-1].shape[1],gpImg2list[i-1].shape[0]))
        lpImg2.append(np.subtract(gpImg2list[i-1],tmp))
        cv2.imshow("img2_laplacian",tmp)
        cv2.waitKey(0)
        
    laplacianList = []
    
    #라플라시안 합 이미지 생성
    for lImg1,lImg2 in zip(lpImg1,lpImg2):
        rows,cols = lImg1.shape
        # Merging first and second half of first and second images respectively at each level in pyramid

        mask1_and=cv2.bitwise_and(mask1,mask2)
        mask1_xor=cv2.bitwise_xor(mask1,mask1_and)
        
        mask1_xor = cv2.resize(mask1_xor,(cols,rows),interpolation = cv2.INTER_CUBIC)
        mask2_ = cv2.resize(mask2,(cols,rows),interpolation = cv2.INTER_CUBIC)
        
        tmp1 = cv2.bitwise_and(lImg1, mask1_xor)
        tmp2 = cv2.bitwise_and(lImg2, mask2_)
        tmp = tmp1 + tmp2

        laplacianList.append(tmp)
    #복원
    img_out = gaussiansumList[-1]
    for i in range(1,levels+1):# xrange 에서 range로 수정
        img_out = cv2.pyrUp(img_out)   # Upsampling the image and merging with higher resolution level image
        img_out = cv2.resize(img_out, (laplacianList[i].shape[1],laplacianList[i].shape[0]))
        img_out = np.add(img_out, laplacianList[i])
        cv2.imshow("img_out",img_out)
    np.clip(img_out, 0, 255, out=img_out)
    return img_out.astype('uint8')

def Stitch_images(baseImage, warpedImage, warpedImageMask):
    rows,cols = baseImage.shape
    for r in range(0,rows):
        for c in range(0,cols):
            if warpedImageMask[r][c] == 255:
                baseImage[r][c] = warpedImage[r][c]
    return baseImage

def Laplacian_cylindrical_warping(img1, img2, img3):
    
    h,w = img1.shape#이미지 형태(모두 단일하다고 고려)
    
    f = 425 # 초점거리
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # camera calibration
    
    (img1cyl, mask1) = cylindricalWarpImage(img1, K)  
    (img2cyl, mask2) = cylindricalWarpImage(img2, K)
    (img3cyl, mask3) = cylindricalWarpImage(img3, K)
    
    # Add Padding
    img1cyl = cv2.copyMakeBorder(img1cyl,50,50,300,300, cv2.BORDER_CONSTANT)
    mask1_ = cv2.copyMakeBorder(mask1,50,50,300,300, cv2.BORDER_CONSTANT)
    
    # image2의 image1으로의 affine trasform matrix 계
    (M21, pts2, pts1, mask4) = getTransform(img2cyl, img1cyl, 'affine')

    # Transform image2 and mask2 in plane of image1.
    transformedImage2 = cv2.warpAffine(img2cyl, M21, (img1cyl.shape[1],img1cyl.shape[0]))
    transformedMask2 = cv2.warpAffine(mask2, M21, (img1cyl.shape[1],img1cyl.shape[0]))
    
    # Stich image1 and image2 using mask.
    #transformedImage21 = Stitch_images(img1cyl, transformedImage2, transformedMask2)

    transformedImage21_ = Laplacian_blending(img1cyl,transformedImage2,mask1_,transformedMask2)
    cv2.imshow("transformedImage21_",transformedImage21_)
    cv2.waitKey(0)
    # Transformation image3 in plane of image1.
    (M31, pts2, pts1, mask5) = getTransform(img3cyl, img1cyl, 'affine')

    transformedImage3 = cv2.warpAffine(img3cyl, M31, (img1cyl.shape[1],img1cyl.shape[0]))
    transformedMask3 = cv2.warpAffine(mask3, M31, (img1cyl.shape[1],img1cyl.shape[0]))
    #transformedImage31 = Stitch_images(transformedImage21_, transformedImage3, transformedMask3)
    black = np.zeros((img1cyl.shape[1],img1cyl.shape[0]))
    mask21 = cv2.bitwise_or(transformedImage21_,black)
    # Combine both transformed images using Laplacian.
    output_image = Laplacian_blending(transformedImage21_, transformedImage3,mask21,transformedMask3 )

    cv2.imshow("output_image",output_image)
    cv2.waitKey(0)
    """
    output_name = sys.argv[5] + "output_cylindrical_lpb.png"
    cv2.imwrite(output_name, output_image)
    """
    return True

#이미지 입력  
input_image1 = cv2.imread("./input1.png",0)
input_image2 = cv2.imread("./input2.png",0)
input_image3 = cv2.imread("./input3.png",0) 

# 원기둥 좌표계 상의 라플라시안 블렌딩 스티칭 함
Laplacian_cylindrical_warping(input_image1, input_image2, input_image3)