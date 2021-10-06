// OpenCV imports
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

// C++ imports
#include <iostream>

// namespaces
using namespace std;
using namespace cv;
#define PI 3.1415926

//frame 가로 세로 그때마다 다르게 입력
int frameWidth = 640;
int frameHeight = 480;

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
			rightSmallestLevel.mul(Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid.back());
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

int main() {

	// capture object
	VideoCapture capture("test_left_1m.mp4");
	VideoCapture capture2("test_right_1m.mp4");

	// mat container to receive images
	Mat source, destination;
	Mat source2, destination2;

	// check if capture was successful
	if (!capture.isOpened()) throw "Error reading video";
	if (!capture2.isOpened()) throw "Error reading video";

	int alpha_ = 90, beta_ = 90, gamma_ = 90, dx_ = 0, dy_ = 0;
	int f_ = 500, dist_ = 500;
	int bl_=2.25;

	namedWindow("Result", 1);

	createTrackbar("Alpha", "Result", &alpha_, 180);
	createTrackbar("Beta", "Result", &beta_, 180);
	createTrackbar("Gamma", "Result", &gamma_, 180);
	createTrackbar("dx", "Result", &dx_, 180);
	createTrackbar("dy", "Result", &dy_, 180);
	//createTrackbar("blend", "Result", &bl_, 10);
	//createTrackbar("f", "Result", &f_, 2000);
	//createTrackbar("Distance", "Result", &dist_ , 2000);
	
	int alpha_2 = 90, beta_2 = 90, gamma_2 = 90, dx_2 = 0, dy_2 = 0;
	int f_2 = 500, dist_2 = 500;

	namedWindow("Result2", 1);

	createTrackbar("Alpha2", "Result2", &alpha_2, 180);
	createTrackbar("Beta2", "Result2", &beta_2, 180);
	createTrackbar("Gamma2", "Result2", &gamma_2, 180);
	createTrackbar("dx2", "Result2", &dx_2, 640);
	createTrackbar("dy2", "Result2", &dy_2, 180);
	//createTrackbar("f2", "Result2", &f_2, 2000);
	//createTrackbar("Distance2", "Result2", &dist_2, 2000);

	while (true) {

		capture >> source;
		capture2 >> source2;

		resize(source, source, Size(frameWidth, frameHeight));
		resize(source2, source2, Size(frameWidth, frameHeight));

		double focalLength, dist, alpha, beta, gamma;
		double focalLength2, dist2, alpha2, beta2, gamma2;

		double dx, dy;
		double dx2, dy2;

		double bl;

		alpha = ((double)alpha_ - 90) * PI / 180;
		beta = ((double)beta_ - 90) * PI / 180;
		gamma = ((double)gamma_ - 90) * PI / 180;
		focalLength = (double)f_;
		dist = (double)dist_;
		dx = (double)dx_;
		dy = (double)dy_;
		bl = (double)bl_;

		alpha2 = ((double)alpha_2 - 90) * PI / 180;
		beta2 = ((double)beta_2 - 90) * PI / 180;
		gamma2 = ((double)gamma_2 - 90) * PI / 180;
		focalLength2 = (double)f_2;
		dist2 = (double)dist_2;
		dx2 = (double)dx_2;
		dy2 = (double)dy_2;

		Size image_size = source.size();
		double w = (double)image_size.width, h = (double)image_size.height;
		
		Size image_size2 = source2.size();
		double w2 = (double)image_size2.width, h2 = (double)image_size2.height;


		// Projecion matrix 2D -> 3D
		Mat A1 = (Mat_<float>(4, 3) <<
			1, 0, -w / 2,
			0, 1, -h / 2,
			0, 0, 0,
			0, 0, 1);
		// Projecion matrix 2D -> 3D
		Mat A12 = (Mat_<float>(4, 3) <<
			1, 0, -w2 / 2,
			0, 1, -h2 / 2,
			0, 0, 0,
			0, 0, 1);


		// Rotation matrices Rx, Ry, Rz

		Mat RX = (Mat_<float>(4, 4) <<
			1, 0, 0, 0,
			0, cos(alpha), -sin(alpha), 0,
			0, sin(alpha), cos(alpha), 0,
			0, 0, 0, 1);

		Mat RY = (Mat_<float>(4, 4) <<
			cos(beta), 0, -sin(beta), 0,
			0, 1, 0, 0,
			sin(beta), 0, cos(beta), 0,
			0, 0, 0, 1);

		Mat RZ = (Mat_<float>(4, 4) <<
			cos(gamma), -sin(gamma), 0, 0,
			sin(gamma), cos(gamma), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
		
		// Rotation matrices Rx, Ry, Rz

		Mat RX2 = (Mat_<float>(4, 4) <<
			1, 0, 0, 0,
			0, cos(alpha2), -sin(alpha2), 0,
			0, sin(alpha2), cos(alpha2), 0,
			0, 0, 0, 1);

		Mat RY2 = (Mat_<float>(4, 4) <<
			cos(beta2), 0, -sin(beta2), 0,
			0, 1, 0, 0,
			sin(beta2), 0, cos(beta2), 0,
			0, 0, 0, 1);

		Mat RZ2 = (Mat_<float>(4, 4) <<
			cos(gamma2), -sin(gamma2), 0, 0,
			sin(gamma2), cos(gamma2), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);


		// R - rotation matrix
		Mat R = RX * RY * RZ;
		
		// R - rotation matrix
		Mat R2 = RX2 * RY2 * RZ2;

		// T - translation matrix
		Mat T = (Mat_<float>(4, 4) <<
			1, 0, 0, dy,
			0, 1, 0, dx,
			0, 0, 1, dist,
			0, 0, 0, 1);
		
		// T - translation matrix
		Mat T2 = (Mat_<float>(4, 4) <<
			1, 0, 0, dx2,
			0, 1, 0, dy2,
			0, 0, 1, dist2,
			0, 0, 0, 1);

		// K - intrinsic matrix 
		Mat K = (Mat_<float>(3, 4) <<
			focalLength, 0, w / 2, 0,
			0, focalLength, h / 2, 0,
			0, 0, 1, 0
			);
		
		// K - intrinsic matrix 
		Mat K2 = (Mat_<float>(3, 4) <<
			focalLength2, 0, w2 / 2, 0,
			0, focalLength2, h2 / 2, 0,
			0, 0, 1, 0
			);

		Mat transformationMat = K * (T * (R * A1));
		Mat transformationMat2 = K2 * (T2 * (R2 * A12));

		//warpPerspective(source, destination, transformationMat, image_size, INTER_CUBIC | WARP_INVERSE_MAP);
		//warpPerspective(source2, destination2, transformationMat2, image_size2, INTER_CUBIC | WARP_INVERSE_MAP);
		warpPerspective(source, destination, transformationMat, Size(source.cols*2, source.rows*1)); //  Size(imgR.cols * 2, imgR.rows * 1)
		warpPerspective(source2, destination2, transformationMat2, Size(source2.cols * 2, source2.rows * 1));//평행이동 추가해야한다

		Mat_<Vec3f> l;
		destination.convertTo(l, CV_32F, 1.0 / 255.0);// 1/255 곱하여 normalize (?)
		Mat_<Vec3f> r;
		destination2.convertTo(r, CV_32F, 1.0 / 255.0);

		Mat_<float> m(destination.rows, destination.cols, 0.0); // 이미지와 동일한 크기의 마스크 정의
		m(Range::all(), Range(0, m.cols / bl)) = 1;//흑백 마스크 (이유는 아직)

		Mat_<Vec3f> blend = LaplacianBlend(l, r, m);//LaplacianBlend하여 blend 결과물 #2
		//cv::Mat crop = blend(cv::Rect(0, 0, 1000, 360));//cropping

		Mat3b blend_8UC3;
		//Mat3b crop_8UC3;
		blend.convertTo(blend_8UC3, CV_8UC3, 255);
		//count++;
		// string filename = cv::format("frame%05d.jpg", count);
		 //imwrite(filename, crop_8UC3); 
		
		imshow("Result", destination);
		imshow("Result2", destination2);
		imshow("Result3", blend); // 결과를 보여준다
		//if (waitKey(10) == 27)
			//break;
		waitKey(100);
	}
	return 0;
}