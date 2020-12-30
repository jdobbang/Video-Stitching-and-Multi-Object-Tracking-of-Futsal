#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/videoio.hpp>
#include <vector>

using namespace std;
using namespace cv;

class LaplacianBlending {
private://선언부
    Mat_<Vec3f> left;
    Mat_<Vec3f> right;
    Mat_<float> blendMask;

    vector<Mat_<Vec3f> > leftLapPyr, rightLapPyr, resultLapPyr;
    Mat leftSmallestLevel, rightSmallestLevel, resultSmallestLevel;
    vector<Mat_<Vec3f> > maskGaussianPyramid; //"masks" are 3-channels for easier multiplication with RGB

    int levels;
//#4
    void buildPyramids() {//left,right 이미지에 대한 라플라시안 피라미드 구축
        buildLaplacianPyramid(left, leftLapPyr, leftSmallestLevel);//#4-1
        buildLaplacianPyramid(right, rightLapPyr, rightSmallestLevel);
        buildGaussianPyramid();//#4-2
    }

    void buildGaussianPyramid() {//마스크의 가우시안 피라미드 구축하기

        assert(leftLapPyr.size() > 0);// leftLapPyr의 입력이 제대로 되었는지!?

        maskGaussianPyramid.clear();//마스크에 대한 가우시안 피라미드 집합 벡터 비워놓기
        Mat currentImg;
        cvtColor(blendMask, currentImg, COLOR_GRAY2BGR);// blend mask >> currentImg로 저장하며 컬러로 변환
        maskGaussianPyramid.push_back(currentImg); //highest level

        currentImg = blendMask;
        for (int l = 1; l < levels + 1; l++) {//level의 개수만큼
            Mat _down;
            if (leftLapPyr.size() > l) {// 해당 level보다 사이즈가 크다면
                pyrDown(currentImg, _down, leftLapPyr[l].size());//downsmapling하여 _down에 저장
            }
            else {//더 작다면
                pyrDown(currentImg, _down, leftSmallestLevel.size()); //smallest level
            }

            Mat down;
            cvtColor(_down, down, COLOR_GRAY2BGR);
            maskGaussianPyramid.push_back(down);
            currentImg = _down;
        }
    }
    //#4-1 한 이미지씩 피라미드 빌드
    void buildLaplacianPyramid(const Mat& img, vector<Mat_<Vec3f> >& lapPyr, Mat& smallestLevel) {
        lapPyr.clear();//피라미드 빌드전 clear
        Mat currentImg = img;
        for (int l = 0; l < levels; l++) {// 미리 입력한 level만큼
            Mat down, up;
            pyrDown(currentImg, down);//downsampling @size
            pyrUp(down, up, currentImg.size());//upsampling @size
            Mat lap = currentImg - up;//입력 이미지에 대해 upsampling image를 빼 lap에 저장
            lapPyr.push_back(lap);//lap을 피라미드
            currentImg = down;//downsmapling을 currentImg
        }
        currentImg.copyTo(smallestLevel);//currentImg(downsmapling)를 smallestLevel에 input
    }

    Mat_<Vec3f> reconstructImgFromLapPyramid() {//#6
        Mat currentImg = resultSmallestLevel;
        for (int l = levels - 1; l >= 0; l--) {
            Mat up;

            pyrUp(currentImg, up, resultLapPyr[l].size());
            currentImg = up + resultLapPyr[l];
        }
        return currentImg;
    }

//#5
    void blendLapPyrs() {
        resultSmallestLevel = leftSmallestLevel.mul(maskGaussianPyramid.back()) +
            rightSmallestLevel.mul(Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid .back());
        for (int l = 0; l < levels; l++) {
            Mat A = leftLapPyr[l].mul(maskGaussianPyramid[l]);
            Mat antiMask = Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid[l];
            Mat B = rightLapPyr[l].mul(antiMask);
            Mat_<Vec3f> blendedLevel = A + B;

            resultLapPyr.push_back(blendedLevel);
        }
    }

//#3
public:
    LaplacianBlending(const Mat_<Vec3f>& _left, const Mat_<Vec3f>& _right, const Mat_<float>& _blendMask, int _levels) :
        left(_left), right(_right), blendMask(_blendMask), levels(_levels)
    {
        assert(_left.size() == _right.size());// 두 이미지 및 마스크의 사이즈 동일하다는 조건
        assert(_left.size() == _blendMask.size());
        buildPyramids();//#4
        blendLapPyrs();//#5
    };

    Mat_<Vec3f> blend() {//#6
        return reconstructImgFromLapPyramid();
    }
};

//#2
Mat_<Vec3f> LaplacianBlend(const Mat_<Vec3f>& l, const Mat_<Vec3f>& r, const Mat_<float>& m) {
    LaplacianBlending lb(l, r, m, 4);//왼쪽 이미지, 오른쪽 이미지, 마스크, level=4 순으로 입력 #3
    return lb.blend();
}

void warpPerspectivePadded(
    const Mat& src, const Mat& dst, const Mat& M, // input matrices
    Mat& src_warped, Mat& dst_padded,
    int flags, int borderMode, const Scalar& borderValue) // OpenCV params
{
    
    Mat transf = M / M.at<double>(2, 2); // ensure a legal homography
    if (flags == WARP_INVERSE_MAP ||
        flags == INTER_LINEAR + WARP_INVERSE_MAP ||
        flags == INTER_NEAREST + WARP_INVERSE_MAP)
    {
        invert(transf, transf);
        flags -= WARP_INVERSE_MAP;
    }

    // it is enough to find where the corners of the image go to find
    // the padding bounds; points in clockwise order from origin
    int src_h = src.rows;
    int src_w = src.cols;
    Vec<Point2f, 4> init_pts, transf_pts;
    init_pts[0] = Point2f(0, 0);
    init_pts[1] = Point2f(src_w, 0);
    init_pts[2] = Point2f(src_w, src_h);
    init_pts[3] = Point2f(0, src_h);
    perspectiveTransform(init_pts, transf_pts, transf);

    // find min and max points
    int min_x, min_y, max_x, max_y;
    min_x = floor(min(
        min(transf_pts[0].x, transf_pts[1].x),
        min(transf_pts[2].x, transf_pts[3].x)));
    min_y = floor(min(
        min(transf_pts[0].y, transf_pts[1].y),
        min(transf_pts[2].y, transf_pts[3].y)));
    max_x = ceil(max(
        max(transf_pts[0].x, transf_pts[1].x),
        max(transf_pts[2].x, transf_pts[3].x)));
    max_y = ceil(max(
        max(transf_pts[0].y, transf_pts[1].y),
        max(transf_pts[2].y, transf_pts[3].y)));
      
    // add transl ation to transformation matrix to shift to positive values
    int anchor_x, anchor_y = 0;
    Mat transl_transf = Mat::eye(3, 3, CV_64F);
    if (min_x < 0)
    {
        anchor_x = -min_x;
        transl_transf.at<double>(0, 2) += anchor_x;
    }
    if (min_y < 0)
    {
        anchor_y = -min_y;
        transl_transf.at<double>(1, 2) += anchor_y;
    }
 
    transf = transl_transf * transf;
    transf /= transf.at<double>(2, 2);

    // create padded destination image
    int dst_h = dst.rows;
    int dst_w = dst.cols;
    int pad_top = anchor_y;
    int pad_bot = max(max_y, dst_h) - dst_h;
    int pad_left = anchor_x;
    int pad_right = max(max_x, dst_w) - dst_w;
    copyMakeBorder(dst, dst_padded, pad_top, pad_bot, pad_left, pad_right,
        borderMode, borderValue);

    // transform src into larger window
    int dst_pad_h = dst_padded.rows;
    int dst_pad_w = dst_padded.cols;
    warpPerspective(src, src_warped, transf, Size(dst_pad_w*2, dst_pad_h),
        flags, borderMode, borderValue);
}

int main() {

    VideoCapture video1("walkingR.mp4");
    VideoCapture video2("walkingL.mp4");

    Mat H2(3, 3, CV_64F);
    double data3[] = { 1,0,0,0,1,0,0,0,1 };
    Mat I(3, 3, CV_64F, data3);

    if (!video1.isOpened())
    {
        cout << "video1 is not opened" << endl;
        return 0;
    }
    if (!video2.isOpened())
    {
        cout << "video2 is not opened" << endl;
        return 0;
    }

    int v_width = video1.get(cv::CAP_PROP_FRAME_WIDTH);
    int v_height = video1.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frames_count1 = video1.get(cv::CAP_PROP_FRAME_COUNT);
    int frames_count2 = video2.get(cv::CAP_PROP_FRAME_COUNT);

    int i = 0;
    while(true) // 딱 한번만 homography 구한다
    {

        UMat imgL, imgR;
        video1 >> imgL;
        video2 >> imgR;

        UMat frame1_gray, frame2_gray;
        cvtColor(imgL, frame1_gray, COLOR_BGR2GRAY);
        cvtColor(imgR, frame2_gray, COLOR_BGR2GRAY);

	    Size size(imgL.cols / 3,imgL.rows / 3);
	    resize(imgL, imgL, size);
	    resize(imgR, imgR, size);
   
        Ptr<DescriptorMatcher> matcher;//특징점 descriptor 매칭을 위한 DescriptorMatcher 클래스의 포인터 변수

        matcher = DescriptorMatcher::create("FlannBased"); // FlannBased 매칭 알고리즘을 사용한 matcher  

        Ptr<SIFT> ext = SIFT::create();//SIFT 클래스를 활용하여 추출할 특징점 포인터 변수 ext
        vector<KeyPoint> kp1, kp2;//img1,img2의 특징점 
        Mat desc1, desc2;//kp1,kp2의 descriptor를 담는 Mat 변수

        ext->detectAndCompute(imgL, Mat(), kp1, desc1);//img1에 대한 특징점들(kp1), descriptor(desc1) 계산
        ext->detectAndCompute(imgR, Mat(), kp2, desc2);//img2에 대한 특징점들, descriptor 계산

        //매칭 및 그리기
        vector<vector<DMatch> > rawMatches;// DMatch 클래스의 변수 rawMatches 
        vector<Point2f> p1, p2;//img1,2의 knn 알고리즘과 ratio test 후의 매칭쌍의 특징점들
        vector<float> distances;//특징점 간의 직선 거리 
        matcher->knnMatch(desc1, desc2, rawMatches, 2);//k==2개의 특징점 매칭에 대하여 matcher(FLANN based 알고리즘 활용)에 대입

        //filter_matches
        for (size_t i = 0; i < rawMatches.size(); i++) {//knn을 통해 선별한 match 쌍들에 대해 matching된 개수 만큼 loop
            const vector<DMatch>& m = rawMatches[i];//각 매칭(rawmatches[i]에 대해 DMatch 클래스의 포인터 변수 지정
            if (m.size() == 2 && m[0].distance < m[1].distance * 0.7)// k==2개의 매칭쌍에 대해 ratio(=0.75) test를 통과한 매치들에 대해 특징점 및 distance 값 push_back
            {
                p1.push_back(kp1[m[0].queryIdx].pt);//매칭된 쌍의 img1 keypoints
                p2.push_back(kp2[m[0].trainIdx].pt);//매칭된 쌍의 img2 keypoints
                //distances.push_back(m[0].distance);//매칭된 쌍의 distance
            }
        }

        vector<uchar> status;
        vector<pair<Point2f, Point2f> > pointPairs;//특징점 쌍

       // Mat H = findHomography(p1, p2, status, RANSAC);// 특징점 쌍과 RANSAC알고리즘을 사용하여 이미지간의 변환행렬 H 도출
        H2 = findHomography(p2, p1, status, RANSAC);
        // 첫번째 프레임에 대해 변환 행렬 한개만 얻어내면 된다!
       break;
    }
    int count=0;
    while (true) {

        UMat imgL, imgR;
        video1 >> imgL;
        video2 >> imgR;

        if (imgL.empty() || imgR.empty())
            break;

        Size size2(imgL.cols / 3, imgL.rows / 3);
        resize(imgL, imgL, size2);
        resize(imgR, imgR, size2);

        //Mask of iamge to be combined so you can get resulting mask
        Mat mask1(imgL.size(), CV_8UC1, Scalar::all(255));
        Mat mask2(imgR.size(), CV_8UC1, Scalar::all(0));
        Mat image1Updated, image2Updated;

        //Warp the masks and the images to their new posistions so their are of all the same  size to be overlayed and blended
        warpPerspective(imgL, image1Updated, I, Size(imgR.cols * 2, imgR.rows * 1));
        warpPerspective(imgR, image2Updated, H2, Size(imgR.cols * 2, imgR.rows * 1));
        //warpPerspective(mask1, mask1, I, Size(imgR.cols * 2, imgR.rows * 1), INTER_LINEAR, BORDER_CONSTANT, 0);
        //warpPerspective(mask2, mask2, H2, Size(imgR.cols * 2, imgR.rows * 1), INTER_LINEAR, BORDER_TRANSPARENT, 255);

        imshow("imgL", imgL);
        imshow("imgR", imgR);

        Mat image1uc(image1Updated.rows, image1Updated.cols, CV_32F);
        Mat image2uc(image2Updated.rows, image2Updated.cols, CV_32F);
        image1uc = image1Updated;
        image2uc = image2Updated;

        Mat_<Vec3f> l;
        image1uc.convertTo(l, CV_32F, 1.0 / 255.0);// 1/255 곱하여 normalize (?)
        Mat_<Vec3f> r;
        image2uc.convertTo(r, CV_32F, 1.0 / 255.0);

        Mat_<float> m(image1uc.rows, image2uc.cols, 0.0); // 이미지와 동일한 크기의 마스크 정의
        m(Range::all(), Range(0, m.cols / 2.25)) = 1;//흑백 마스크 (이유는 아직)

        Mat_<Vec3f> blend = LaplacianBlend(l, r, m);//LaplacianBlend하여 blend 결과물 #2
        cv::Mat crop = blend(cv::Rect(0, 0, 1000, 360));//cropping

        Mat3b crop_8UC3;
        crop.convertTo(crop_8UC3, CV_8UC3, 255);
        count++;
       // string filename = cv::format("frame%05d.jpg", count);
        //imwrite(filename, crop_8UC3); 
       imshow("Result", crop); // 결과를 보여준다
       if (waitKey(10)==27) 
            break;
    }
    return 0;
}
