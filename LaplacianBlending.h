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
	Mat_<Vec3f> left; // ���� ������
	Mat_<Vec3f> right; // ������ ������
	Mat_<float> blendMask; // multi-band blending ����ũ

	vector<Mat_<Vec3f> > leftLapPyr, rightLapPyr, resultLapPyr;//��,�� �����ӿ� ���� ���ö�þ� �Ƕ�̵�� ����ġ ���� ���ö�þ� �Ƕ�̵�
	Mat leftSmallestLevel, rightSmallestLevel, resultSmallestLevel; // lowest frequency ���ö�þ� �̹���
	vector<Mat_<Vec3f> > maskGaussianPyramid; // ����þ� �Ƕ�̵�(���� ����ũ)

	int levels;


	void buildPyramids() {
		buildLaplacianPyramid(left, leftLapPyr, leftSmallestLevel);//�� �̹����� ���� ���ö�þ� �Ƕ�̵�
		buildLaplacianPyramid(right, rightLapPyr, rightSmallestLevel);//�� �̹����� ���� ���ö�þ� �Ƕ�̵�
		buildGaussianPyramid();// ���� ����ũ �Ƕ�̵�
	}

	//����þ� �Ƕ�̵� ����
	void buildGaussianPyramid();

	//�� �����Ӿ� �̹����� �Ƕ�̵� ����
	void buildLaplacianPyramid(const Mat& img, vector<Mat_<Vec3f> >& lapPyr, Mat& smallestLevel);

	//�¿� ���ö�þ� �Ƕ�̵� �� ���� ���� ����
	Mat_<Vec3f> reconstructImgFromLapPyramid();

	//���յ� ���ö�þ� �Ƕ�̵� ����
	void blendLapPyrs();


public://�� ������ ,�� ������ , blendmask( ���� ���� �� ������ν� �ʿ��� ����ũ), ��� ���� 
	LaplacianBlending(const Mat_<Vec3f>& _left, const Mat_<Vec3f>& _right, const Mat_<float>& _blendMask, int _levels) :

		left(_left), right(_right), blendMask(_blendMask), levels(_levels)
	{
		assert(_left.size() == _right.size());// �� �̹��� �� ����ũ�� ������ ���� ����
		assert(_left.size() == _blendMask.size());
		buildPyramids();// ����þ� �� ���ö�þ� �Ƕ�̵� ����
		blendLapPyrs();// ���յ� ���ö�þ� �Ƕ�̵� ����
	};

	Mat_<Vec3f> blend() { return reconstructImgFromLapPyramid(); }
};

