#include "LaplacianBlending.h"


void LaplacianBlending::buildLaplacianPyramid(const Mat& img, vector<Mat_<Vec3f> >& lapPyr, Mat& smallestLevel)
{
	lapPyr.clear();
	Mat currentImg = img;
	for (int l = 0; l < levels; l++) {// 사용자가 지정한 밴드 level
		Mat down, up;
		pyrDown(currentImg, down);//프레임에 대해 Down 샘플링
		pyrUp(down, up, currentImg.size());// 다운 샘플링에 대한 UP 샘플링
		Mat lap = currentImg - up;// Down 샘플링 - Up 샘플링 = 라플라시안 이미지
		lapPyr.push_back(lap);//라플라시안 피라미드 생성
		currentImg = down;//다음 level의 Down 샘플링을 위해
	}
	currentImg.copyTo(smallestLevel);//마지막 가장 작은 downSampling의 currentImg은 smallestLevel에 저장
}

void LaplacianBlending::buildGaussianPyramid() {//가우시안 피라미드 빌드

	assert(leftLapPyr.size() > 0);
	maskGaussianPyramid.clear();

	Mat currentImg;
	cvtColor(blendMask, currentImg, COLOR_GRAY2BGR);// 컬러 프레임에 대한 이미지 블렌딩용 배경 마스크

	maskGaussianPyramid.push_back(currentImg); //highest level
	currentImg = blendMask;
	for (int l = 1; l < levels + 1; l++) {//band level
		Mat _down;
		if (leftLapPyr.size() > l) {// 라플라시안 피라미드의 사이즈가 'I'보다 크다면 이미지 라플라시안 피라미드의 'I'번째 레벨의 사이즈로 마스크를 downsmapling
			pyrDown(currentImg, _down, leftLapPyr[l].size());
		}
		else {//아니라면 smallest level 크기로 다운샘플링
			pyrDown(currentImg, _down, leftSmallestLevel.size());
		}

		Mat down;
		cvtColor(_down, down, COLOR_GRAY2BGR);
		maskGaussianPyramid.push_back(down); // 마스크의 가우시안 피라미드 생성
		currentImg = _down;
	}
}

void LaplacianBlending::blendLapPyrs()
{	//lowest frequency의 좌, 우 라플라시안 이미지를 가우시안 마스크를 이용해 블렌딩(공식)
	resultSmallestLevel = leftSmallestLevel.mul(maskGaussianPyramid.back()) + rightSmallestLevel.mul(Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid.back());

	for (int l = 0; l < levels; l++) {// 각 밴드 레벨 별로
		Mat A = leftLapPyr[l].mul(maskGaussianPyramid[l]);
		Mat antiMask = Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid[l];
		Mat B = rightLapPyr[l].mul(antiMask);
		Mat_<Vec3f> blendedLevel = A + B;

		resultLapPyr.push_back(blendedLevel); // 공식대로 결합된 라플라시안 이미지 피라미드
	}
}

Mat_<Vec3f> LaplacianBlending::reconstructImgFromLapPyramid() {//최종 이미지 만들기  
	Mat currentImg = resultSmallestLevel;
	for (int l = levels - 1; l >= 0; l--) {
		Mat up;
		pyrUp(currentImg, up, resultLapPyr[l].size());
		currentImg = up + resultLapPyr[l];// low frequency를 high freuquency로 사이즈를 맞춰 gradually(level by level) 블렌딩
	}
	return currentImg;// 최종 결과물
}

