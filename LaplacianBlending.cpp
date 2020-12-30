#include "LaplacianBlending.h"


void LaplacianBlending::buildLaplacianPyramid(const Mat& img, vector<Mat_<Vec3f> >& lapPyr, Mat& smallestLevel)
{
	lapPyr.clear();
	Mat currentImg = img;
	for (int l = 0; l < levels; l++) {// ����ڰ� ������ ��� level
		Mat down, up;
		pyrDown(currentImg, down);//�����ӿ� ���� Down ���ø�
		pyrUp(down, up, currentImg.size());// �ٿ� ���ø��� ���� UP ���ø�
		Mat lap = currentImg - up;// Down ���ø� - Up ���ø� = ���ö�þ� �̹���
		lapPyr.push_back(lap);//���ö�þ� �Ƕ�̵� ����
		currentImg = down;//���� level�� Down ���ø��� ����
	}
	currentImg.copyTo(smallestLevel);//������ ���� ���� downSampling�� currentImg�� smallestLevel�� ����
}

void LaplacianBlending::buildGaussianPyramid() {//����þ� �Ƕ�̵� ����

	assert(leftLapPyr.size() > 0);
	maskGaussianPyramid.clear();

	Mat currentImg;
	cvtColor(blendMask, currentImg, COLOR_GRAY2BGR);// �÷� �����ӿ� ���� �̹��� ������ ��� ����ũ

	maskGaussianPyramid.push_back(currentImg); //highest level
	currentImg = blendMask;
	for (int l = 1; l < levels + 1; l++) {//band level
		Mat _down;
		if (leftLapPyr.size() > l) {// ���ö�þ� �Ƕ�̵��� ����� 'I'���� ũ�ٸ� �̹��� ���ö�þ� �Ƕ�̵��� 'I'��° ������ ������� ����ũ�� downsmapling
			pyrDown(currentImg, _down, leftLapPyr[l].size());
		}
		else {//�ƴ϶�� smallest level ũ��� �ٿ���ø�
			pyrDown(currentImg, _down, leftSmallestLevel.size());
		}

		Mat down;
		cvtColor(_down, down, COLOR_GRAY2BGR);
		maskGaussianPyramid.push_back(down); // ����ũ�� ����þ� �Ƕ�̵� ����
		currentImg = _down;
	}
}

void LaplacianBlending::blendLapPyrs()
{	//lowest frequency�� ��, �� ���ö�þ� �̹����� ����þ� ����ũ�� �̿��� ����(����)
	resultSmallestLevel = leftSmallestLevel.mul(maskGaussianPyramid.back()) + rightSmallestLevel.mul(Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid.back());

	for (int l = 0; l < levels; l++) {// �� ��� ���� ����
		Mat A = leftLapPyr[l].mul(maskGaussianPyramid[l]);
		Mat antiMask = Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid[l];
		Mat B = rightLapPyr[l].mul(antiMask);
		Mat_<Vec3f> blendedLevel = A + B;

		resultLapPyr.push_back(blendedLevel); // ���Ĵ�� ���յ� ���ö�þ� �̹��� �Ƕ�̵�
	}
}

Mat_<Vec3f> LaplacianBlending::reconstructImgFromLapPyramid() {//���� �̹��� �����  
	Mat currentImg = resultSmallestLevel;
	for (int l = levels - 1; l >= 0; l--) {
		Mat up;
		pyrUp(currentImg, up, resultLapPyr[l].size());
		currentImg = up + resultLapPyr[l];// low frequency�� high freuquency�� ����� ���� gradually(level by level) ����
	}
	return currentImg;// ���� �����
}

