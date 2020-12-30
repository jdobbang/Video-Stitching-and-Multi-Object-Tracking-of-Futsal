#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
//opencv에서 제공하는 교육용 SURF사용(라이선스때문)
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "LaplacianBlending.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const int LOOP_NUM = 10;
const int GOOD_PTS_MAX = 50;
const float GOOD_PORTION = 0.15f;

///SURF를 사용한 feature 검출기
struct SURFDetector 
{
	Ptr<Feature2D> surf;
	SURFDetector(double hessian = 800.0)
	{
		surf = SURF::create(hessian);
	}

	template<class Type>
	void operator()(const Type& in, const Type& mask, vector<cv::KeyPoint>& pts, Type& descriptors, bool useProvided = false)
	{
		//여기서 detect랑 descriptor생성을 동시에 진행
		surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
	}
};

template<class KPMatcher>
struct SURFMatcher
{
	KPMatcher matcher;

	template<class T>
	void match(const T& in1, const T& in2, vector<DMatch>& matches) 
	{
		matcher.match(in1, in2, matches);
	}
};


///좋은 매칭을 가지는 feature들을 찾아내는 함수
void getGoodMatches(vector<DMatch>& matches, vector< DMatch >& good_matches,
	const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, vector<Point2f>& img_1, vector<Point2f>& img_2)
{
	sort(matches.begin(), matches.end());

	//GOOD_PORTION%의 좋은 특징점들을 저장
	const int ptsPairs = min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));
	for (int i = 0; i < ptsPairs; i++)
		good_matches.push_back(matches[i]);

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//good_match로부터 good features 얻어옴
		img_1.push_back(keypoints1[good_matches[i].queryIdx].pt);
		img_2.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}
}

///좋은 매칭을 가지는 특징점들을 화면에 출력하는 함수
static Mat drawGoodMatches(const Mat& img1, const Mat& img2, 
	const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
	vector<DMatch>& good_matches, vector<Point2f>& img2_corners_, Mat& H)
{
	//img1에서 corner를 얻어옴
	vector<Point2f> img1_corners(4);
	img1_corners[0] = Point(0, 0);
	img1_corners[1] = Point(img1.cols, 0);
	img1_corners[2] = Point(img1.cols, img1.rows);
	img1_corners[3] = Point(0, img1.rows);
	
	vector<Point2f> img2_corners(4);
	
	Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//Draw lines between the corners
	img2_corners = img2_corners_;
	perspectiveTransform(img2_corners, img1_corners, H);

	for (int i = 0; i < 4; i++)
	{
		line(img_matches,
			img2_corners[i] + Point2f((float)img1.cols, 0), img2_corners[(i + 1) % 4] + Point2f((float)img1.cols, 0),
			Scalar(0, 255, 0), 2, LINE_AA);
	}

	return img_matches;
}

///src 이미지의 모든 특징점을 print
static Mat drawAllKeypoints(const Mat& src, const vector<KeyPoint>& srcKeypoints)
{
	Mat printKeyPoint;
	drawKeypoints(src, srcKeypoints, printKeyPoint, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	return printKeyPoint;
}

///DLT algorithm을 사용해서 homography를 구하는 함수
Mat getHomography_basicDLT(const vector<Point2f>& src_points, const vector<Point2f>& dst_points) 
{
	int feautreSize = src_points.size();

	//homography연산을 위한 A matrix 구축
	//A matrix: transform 연산을 수행할 때 대응관계에 있는 좌표들로 구해지는 행렬
	Mat A(feautreSize * 2, 9, CV_32FC1);
	for (int i = 0; i < feautreSize; i++)
	{
		A.at<float>(2 * i, 0) = -src_points[i].x;
		A.at<float>(2 * i, 1) = -src_points[i].y;
		A.at<float>(2 * i, 2) = -1.0f;
		A.at<float>(2 * i, 3) = 0.0f;
		A.at<float>(2 * i, 4) = 0.0f;
		A.at<float>(2 * i, 5) = 0.0f;
		A.at<float>(2 * i, 6) = dst_points[i].x * src_points[i].x;
		A.at<float>(2 * i, 7) = dst_points[i].x * src_points[i].y;
		A.at<float>(2 * i, 8) = dst_points[i].x;
	
		A.at<float>(2 * i + 1, 0) = 0.0f;
		A.at<float>(2 * i + 1, 1) = 0.0f;
		A.at<float>(2 * i + 1, 2) = 0.0f;
		A.at<float>(2 * i + 1, 3) = -src_points[i].x;
		A.at<float>(2 * i + 1, 4) = -src_points[i].y;
		A.at<float>(2 * i + 1, 5) = -1.0f;
		A.at<float>(2 * i + 1, 6) = dst_points[i].y * src_points[i].x;
		A.at<float>(2 * i + 1, 7) = dst_points[i].y * src_points[i].y;
		A.at<float>(2 * i + 1, 8) = dst_points[i].y;
	}
	Mat u, d, vt, V;	//A = u d vt -> SVD에 의해 3개의 행렬의 곱으로 나타내진다
	SVD::compute(A, d, u, vt, SVD::FULL_UV); //singluar value decomposition을 이용해 vt행렬을 구한다
	transpose(vt, V);
	
	Mat customH(3, 3, V.type());	//matrix vt의 마지막 column vector을 customH로 가지고 온다
	//h_33을 1로 만들어 주기 위해서 마지막값으로 전체 customH 행렬 요소들을 나누어줌
	float lastVal = V.at<float>(8, 8);
	for (int h = 0; h < 3; h++)
		for (int w = 0; w < 3; w++)
		{
			customH.at<float>(h, w) = V.at<float>(h * 3 + w, 8) / lastVal;
		}

	return customH;
}


///특징점들의 좌표들을 정규화시키는 함수(normalized DLT algorithm을 이용해서 homography를 구할 때 사용)
void pointNormalization(const vector<Point2f>& points, vector<Point2f>& normalized_points, Mat& T) 
{
	//data normalization by hartely를 사용
	//1) 좌표들의 centroid들이 origin(0,0)에 위치하게 변환
	int featureSize = points.size();
	float xCenter = 0.0f, yCenter = 0.0f;
	float dist = 0.0f;
	float scaling = 0.0f;	//similarity transform T를 계산하기 위한 scalar

	//centroid 구하기
	for (int i = 0; i < featureSize; i++)
	{
		xCenter += points[i].x;
		yCenter += points[i].y;
	}
	xCenter /= featureSize;
	yCenter /= featureSize;

	for (int i = 0; i < featureSize; i++)
	{
		dist += sqrt(pow(points[i].x - xCenter, 2) + pow(points[i].y - xCenter, 2));
	}
	dist /= featureSize;
	//2) 좌표들은 scale되서 좌표들에서 origin까지 거리의 평균이 sqrt(2)
	//similarity transfrom T생성
	scaling = 1.0f / dist;
	float t[3 * 3] = { scaling,0,-scaling * xCenter,0,scaling, -scaling * yCenter, 0,0,1 };
	memcpy(T.data, t, 3 * 3 * sizeof(float));

	float x, y;
	for (int i = 0; i < featureSize; i++)
	{
		x = points[i].x;
		y = points[i].y;

		//homogenious coordinate로 표현
		normalized_points[i].x = (t[0] * x + t[1] * y + t[2]);
		normalized_points[i].y = (t[3] * x + t[4] * y + t[5]);
	}
}

Mat getHomography_normalizedDLT(const vector<Point2f>& src_points, const vector<Point2f>& dst_points) 
{
	///normalized DLT algorithm을 사용해서 homography를 구함
	//H = inv(Tp) * customH * T
	int featureSize = src_points.size();
	Mat T(3, 3, CV_32FC1);	//similiarity transfromation matrix of src_points
	Mat Tp(3, 3, CV_32FC1);	//similiarity transfromation matrix of dst_points

	vector<Point2f> src_points_norm = src_points;
	vector<Point2f> dst_points_norm = dst_points;
	pointNormalization(src_points, src_points_norm, T);
	pointNormalization(dst_points, dst_points_norm, Tp);

	//정규화한 feature points들에 대해 homography를 구함
	Mat customH = getHomography_basicDLT(src_points_norm, dst_points_norm);

	//H = inv(Tp) * customH * T (denormalization하는 과정)
	Mat invertTp, Htemp, resultH;
	invert(Tp, invertTp);
	multiply(customH, T, Htemp);
	multiply(invertTp, Htemp, resultH);

	return resultH;
}

Mat_<Vec3f> LaplacianBlend(const Mat_<Vec3f>& l, const Mat_<Vec3f>& r, const Mat_<float>& m) {
	LaplacianBlending lb(l, r, m, 4);//왼쪽 이미지, 오른쪽 이미지, 마스크, level=4 순으로 입력
	return lb.blend();
}


int main() 
{
	//video 불러오기
	VideoCapture video1("original_left.mp4");
	VideoCapture video2("original_right.mp4");

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

	cout << "VIDEO WIDTH: " << v_width << endl;
	cout << "VIDEO HEIGTH: " << v_height << endl;
	cout << "VIDEO1/VIDEO2 FRAME COUNT: " << frames_count1 << "/" << frames_count2 << endl;

	//stitching된 결과를 저장하는 videoWriter
	VideoWriter stitched_writer;
	stitched_writer.open("resultVideo.avi", CV_FOURCC('D', 'I', 'V', 'X'), 24, Size(1250, 480), true);
	if (!stitched_writer.isOpened())
	{
		cout << "writer is not opened" << endl;
		return 0;
	}

	vector<KeyPoint> keypoints1, keypoints2;
	vector<DMatch> bf_matches;

	Mat descriptors1, descriptors2;

	SURFDetector surf;
	SURFMatcher<BFMatcher> bf_matcher;
	
	Mat staticHomography_basic(3, 3, CV_64F); // 고정 호모그래피, blending 함수는 64F 데이터 형식으로 호모그래피 전달 
	
	double data3[] = { 1,0,0,0,1,0,0,0,1 }; // 블랜딩 시에 왼쪽 이미지는 항등행렬변환을 하여 그대로 머물러 기준이 되도록 곱한다.
	Mat I(3, 3, CV_64F, data3);

	//영상의 첫 frames에서 고정된 homography를 구함
	Mat frame1, frame2;
	video1 >> frame1;
	video2 >> frame2;

	Mat frame1_gray, frame2_gray;
	cvtColor(frame1, frame1_gray, CV_BGR2GRAY);
	cvtColor(frame2, frame2_gray, CV_BGR2GRAY);

	for (int i = 0; i < LOOP_NUM; i++)
	{
		//surf로 feature detect and descriptor 생성
		surf(frame1_gray, Mat(), keypoints1, descriptors1);
		surf(frame2_gray, Mat(), keypoints2, descriptors2);

		//feature들 matching
		bf_matcher.match(descriptors1, descriptors2, bf_matches);
	}

	vector<DMatch> good_matches;
	vector<Point2f> img_1, img_2; //good features

	getGoodMatches(bf_matches, good_matches, keypoints1, keypoints2, img_1, img_2);

	//DLT algorithm을 사용
	Mat basicH = getHomography_basicDLT(img_2, img_1);
	//normalized DLT algorithm을 사용
	Mat normH = getHomography_normalizedDLT(img_2, img_1);
	
	staticHomography_basic = basicH;

	int cnt = 0; //진행되고 있는 frame 개수 확인
	while (true)
	{
		UMat imgL, imgR; // 컬러 이미지로 변환은 하지 않는다.
		video1 >> imgL;
		video2 >> imgR;

		if (imgL.empty()) break;
		if (imgR.empty()) break;

		++cnt;
		cout << "=================" << cnt << "=================" << endl;

		//이미지와 마스크를 warping, 마스크는 블렌딩 실행 시 배경에서 필요하다.
		Mat image1Updated, image2Updated;
		Mat tempUpdated; //for normalized DLT

		// 왼쪽 이미지는 기준으로 항등행렬 I를 곱해 사이즈만 블렌딩 최종 이미지에 맞게 warp
		warpPerspective(imgL, image1Updated, I, Size(imgR.cols * 2, imgR.rows * 1));
		// 오른쪽 이미지는 staticHomography로 warp한다.
		warpPerspective(imgR, image2Updated, staticHomography_basic, Size(imgR.cols * 2, imgR.rows * 1));
		
		//normalized homography를 적용할 시에 위에 staticHomography_basic을 normH로 바꾸어주고 
		//바로 위에 warpPerspective함수를 주석처리후 아래 주석을 제거
		
		/*warpPerspective(imgR, tempUpdated, staticHomography_basic, Size(imgR.cols * 2, imgR.rows * 1)); 
		float translationData[] = { 1,0,((float)imgR.cols*0.36), 0,1,0 }; //normalized DLT로 구한 homography에 대해 위치 이동
		Mat translationMatrix(2,3, CV_32F, translationData);
		warpAffine(tempUpdated,image2Updated, translationMatrix, Size(imgR.cols * 2, imgR.rows));
		*/

		Mat image1uc(image1Updated.rows, image1Updated.cols, CV_32F);//Laplacain 블렌드를 위해 이미지는 CV_32F로 변형
		Mat image2uc(image2Updated.rows, image2Updated.cols, CV_32F);
		image1uc = image1Updated;
		image2uc = image2Updated;

		Mat_<Vec3f> l;
		image1uc.convertTo(l, CV_32F, 1.0 / 255.0);// 라플라시안 블렌딩을 위해 Normalize
		Mat_<Vec3f> r;
		image2uc.convertTo(r, CV_32F, 1.0 / 255.0);

		Mat_<float> m(image1uc.rows, image2uc.cols, 0.0); // 이미지와 동일한 크기의 마스크 정의
		m(Range::all(), Range(0, m.cols/2.1)) = 1;//블렌딩을 위한 흑백 마스크

		Mat_<Vec3f> blend = LaplacianBlend(l, r, m);//LaplacianBlend의 결과
		Mat crop = blend(Rect(0, 0, 1250, 480));// (0,0)과 (1200,36)의 좌표를 대각으로 갖는 사각형으로 크롭

		Mat3b crop_8UC3;
		crop.convertTo(crop_8UC3, CV_8UC3, 255); // crop을 8UC3 자료형으로 변환시키는 함수(VideoWriter에 저장하기 위해서)

		//videoWriter를 이용해 저장
		stitched_writer.write(crop_8UC3);
	}

	return 0;
}