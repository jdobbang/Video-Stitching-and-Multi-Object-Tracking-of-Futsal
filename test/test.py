import cv2
import sys


img_names = ['./left.png', './right.png']

# 불러온 영상을 imgs에 저장
imgs = []
for name in img_names:
    img = cv2.imread(name)
    img = cv2.resize(img, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR) #일단 임의로 사이즈 조절
    cv2.imshow("img",img)
    cv2.waitKey(0)
    if img is None:
        print('Image load failed!')
        sys.exit()
        
    imgs.append(img)
    
print("stitching start..")
# 객체 생성
stitcher = cv2.Stitcher_create()

print("stitching executing")
# 이미지 스티칭
status, dst = stitcher.stitch(imgs)
print("stitching done")

if status != cv2.Stitcher_OK:
    print('Stitch failed!')
    sys.exit()
    
# 출력 영상이 화면보다 커질 가능성이 있어 WINDOW_NORMAL 지정
cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()
