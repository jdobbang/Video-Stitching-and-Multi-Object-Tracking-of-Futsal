#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

class LaplacianBlending
{
private:
	Mat_<Vec3f> left; // 왼쪽 프레임
	Mat_<Vec3f> right; // 오른쪽 프레임
	Mat_<float> blendMask; // multi-band blending 마스크

	vector<Mat_<Vec3f> > leftLapPyr, rightLapPyr, resultLapPyr;//좌,우 프레임에 대한 라플라시안 피라미드와 가중치 적용 라플라시안 피라미드
	Mat leftSmallestLevel, rightSmallestLevel, resultSmallestLevel; // lowest frequency 라플라시안 이미지
	vector<Mat_<Vec3f> > maskGaussianPyramid; // 가우시안 피라미드(블렌딩 마스크)

	int levels;


	void buildPyramids() {
		buildLaplacianPyramid(left, leftLapPyr, leftSmallestLevel);//좌 이미지에 대한 라플라시안 피라미드
		buildLaplacianPyramid(right, rightLapPyr, rightSmallestLevel);//우 이미지에 대한 라플라시안 피라미드
		buildGaussianPyramid();// 블렌딩 마스크 피라미드
	}

	//가우시안 피라미드 빌드
	void buildGaussianPyramid();

	//한 프레임씩 이미지씩 피라미드 빌드
	void buildLaplacianPyramid(const Mat& img, vector<Mat_<Vec3f> >& lapPyr, Mat& smallestLevel);

	//좌우 라플라시안 피라미드 각 레벨 별로 결합
	Mat_<Vec3f> reconstructImgFromLapPyramid();

	//결합된 라플라시안 피라미드 블렌딩
	void blendLapPyrs();


public://좌 프레임 ,우 프레임 , blendmask( 블렌드 수행 시 배경으로써 필요한 마스크), 밴드 레벨 
	LaplacianBlending(const Mat_<Vec3f>& _left, const Mat_<Vec3f>& _right, const Mat_<float>& _blendMask, int _levels) :

		left(_left), right(_right), blendMask(_blendMask), levels(_levels)
	{
		assert(_left.size() == _right.size());// 두 이미지 및 마스크의 사이즈 동일 조건
		assert(_left.size() == _blendMask.size());
		buildPyramids();// 가우시안 및 라플라시안 피라미드 빌드
		blendLapPyrs();// 결합된 라플라시안 피라미드 블레딩
	};

	Mat_<Vec3f> blend() { return reconstructImgFromLapPyramid(); }
};

