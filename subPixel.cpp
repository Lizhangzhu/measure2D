#include"subPixel.h"

using namespace std;
using namespace cv;

void subPixelBaseGauss(const Mat& src, vector<Point2f>& subPixelData ,const double thres) {
	Mat gauss;

	GaussianBlur(src, gauss, Size(3, 3), 1.5);

	//�����ݶ�ͼ��
	Mat sobel_x, sobel_y;

	Sobel(gauss, sobel_x, CV_16SC1, 1, 0);
	Sobel(gauss, sobel_y, CV_16SC1, 0, 1);
	Mat abs_x = abs(sobel_x);
	Mat abs_y = abs(sobel_y);
	Mat gradient = abs_x + abs_y;

	//����ݶȴ�����ֵ�����ݶȷ����Ǽ���ֵ�����ж�Ϊ���ؼ���Ե��
	int rows = gauss.rows - 2;
	int cols = gauss.cols - 2;

	//��һ������һ�еĵ�ַ���
	ptrdiff_t  GradStep = sobel_x.ptr<short>(1) - sobel_x.ptr<short>(0);
	//���ؼ���Ե�������
	vector<Point2i> edgePoints;
		//���Դ���
		//Mat show(gauss.rows, gauss.cols, CV_8UC1,Scalar::all(0));
	for (int r = 2; r < rows; r++)
	{	
		short* pGrad_x = sobel_x.ptr<short>(r);
		short* pGrad_y = sobel_y.ptr<short>(r);
		short* pGradAbs_x = abs_x.ptr<short>(r);
		short* pGradAbs_y = abs_y.ptr<short>(r);
		short* pGradient = gradient.ptr<short>(r);

		for (int c = 2; c < cols; c++) 
		{
			if (pGradient[c] > thres)
			{
				//�ж����ݶȷ������Ƿ��Ǽ���ֵ
#define TG22 0.4142135623730950488016887242097//����������ǰ����������
#define TG67 2.4142135623730950488016887242097
							
				//С��22.5��
				if (pGradAbs_y[c] < TG22 * pGradAbs_x[c])//���������������˷�ĸΪ0�����
				{
					if (pGradient[c] > pGradient[c - 1] && pGradient[c] >= pGradient[c + 1])
					{
						//�������ؼ���Ե��
						edgePoints.emplace_back(c,r);		
						//show.at<uchar>(r, c) = 255;
					}
				}
				//����67.5��
				else if(pGradAbs_y[c] > TG67 * pGradAbs_x[c])
				{
					if (pGradient[c] > pGradient[c - GradStep] && pGradient[c] >= pGradient[c + GradStep])
					{
						//�������ؼ���Ե��
						edgePoints.emplace_back(c, r);
						//show.at<uchar>(r, c) = 255;//���Դ���
					}
				}
				else
				{
					//������������λ����ͳһ����������������Ч�ʡ�dx��dyͬ����Ϊ1�������Ϊ-1��
					int shift = ((pGrad_x[c] ^ pGrad_y[c]) >= 0 ? 1 : -1);
					if (pGradient[c] > pGradient[c - GradStep - shift] && 
						pGradient[c] >= pGradient[c + GradStep + shift])
					{
						///�������ؼ���Ե��
						edgePoints.emplace_back(c, r);
						//show.at<uchar>(r, c) = 255;
					}				
				}
			}
		}
	}

	//�洢���������ݶȷ����ͶӰ���룬�Լ��Ҷ�ֵ��
	struct interpilateData {
		float projection[5];
		int Grayscale[5];
	}interData;

	
	Point2i sample[5];//�ݶȷ��򸽽��ĵ�

	for (int i = 0; i < edgePoints.size(); i++) {
		sample[0] = edgePoints[i];
		//�����ݶȷ���ͼ������ϵΪ��׼��
		float angle360 = fastAtan2(sobel_y.at<short>(sample[0].y, sample[0].x),
			sobel_x.at<short>(sample[0].y, sample[0].x));
		float _angle180 = angle360 > 180 ? (angle360 - 180) : angle360;

		int angle_case = _angle180 / 11.3;//11.25��������

		int posShift_x1 = 0, posShift_y1 = 0;
		int posShift_x2 = 0, posShift_y2 = 0;//x3��y3.x4��y4������1��2�෴��
		
		switch (angle_case)
		{
		case 0:
			posShift_x1 = 1; posShift_y1 = 0;	
			posShift_x2 = 2; posShift_y2 = 0;
			break;
		case 1:
			posShift_x1 = 1; posShift_y1 = 0;	
			posShift_x2 = 2; posShift_y2 = 1;	
			break;
		case 2:
			posShift_x1 = 1; posShift_y1 = 1;
			posShift_x2 = 2; posShift_y2 = 1;	
			break;
		case 3:
		case 4:
			posShift_x1 = 1; posShift_y1 = 1;
			posShift_x2 = 2; posShift_y2 = 2;	
			break;
		case 5:
			posShift_x1 = 1; posShift_y1 = 1;	
			posShift_x2 = 1; posShift_y2 = 2;	
			break;
		case 6:
			posShift_x1 = 0; posShift_y1 = 1;	
			posShift_x2 = 1; posShift_y2 = 2;
			break;
		case 7:
		case 8:
			posShift_x1 = 0; posShift_y1 = 1;	
			posShift_x2 = 0; posShift_y2 = 2;	
			break;
		case 9:
			posShift_x1 = 0;  posShift_y1 = 1;	
			posShift_x2 = -1; posShift_y2 = 2;	
			break;
		case 10:
			posShift_x1 = -1; posShift_y1 = 1;	
			posShift_x2 = -1; posShift_y2 = 2;		
			break;
		case 11:
		case 12:
			posShift_x1 = -1; posShift_y1 = 1;
			posShift_x2 = -2; posShift_y2 = 2;	
			break;
		case 13:
			posShift_x1 = -1; posShift_y1 = 1;	
			posShift_x2 = -2; posShift_y2 = 1;	
			break;
		case 14:
			posShift_x1 = -1; posShift_y1 = 0;		
			posShift_x2 = -2; posShift_y2 = 1;	
			break;
		case 15:
			posShift_x1 = -1; posShift_y1 = 0;		
			posShift_x2 = -2; posShift_y2 = 0;
			break;
		default:
			break;
		}
		sample[1] = Point2i(edgePoints[i].x + posShift_x1, edgePoints[i].y+ posShift_y1);
		sample[2] = Point2i(edgePoints[i].x + posShift_x2, edgePoints[i].y+ posShift_y2);
		sample[3] = Point2i(edgePoints[i].x - posShift_x1, edgePoints[i].y- posShift_y1);
		sample[4] = Point2i(edgePoints[i].x - posShift_x2, edgePoints[i].y- posShift_y2);
		//�����ݶȷ�������
		Point2f vec(sobel_x.at<short>(edgePoints[i].y, edgePoints[i].x), sobel_y.at<short>(edgePoints[i].y, edgePoints[i].x));
		//�涨�ݶȷ���Ϊ������
		//��sample�����ݶȷ����ϵ�ͶӰ������������
		//a*b = |a|*|b|*cos
		for (int _index = 0; _index < 5; _index++) {
			Point2f b(sample[_index].x - edgePoints[i].x, sample[_index].y - edgePoints[i].y);

			interData.projection[_index] = vec.ddot(b) / (sqrt(vec.ddot(vec)));
			interData.Grayscale[_index] = gauss.at<uchar>(sample[_index].y, sample[_index].x) - 10;
		}

		//����ϵ������X
		cv::Mat X(5, 3, CV_32FC1, cv::Scalar(0));
		//�������ֵ����Z
		cv::Mat Y(5, 1, CV_32FC1, cv::Scalar(0));

		for (int X_row = 0; X_row < 5; X_row++) {
			float* ptr = X.ptr<float>(X_row);
			for (int X_col = 0; X_col < 3; X_col++) {
				ptr[X_col] = pow(interData.projection[X_row], X_col);
			}
			Y.at<float>(X_row, 0) = log(interData.Grayscale[X_row]);
		}

		//������A: X*A =Z

		cv::Mat A(3, 1, CV_32FC1, cv::Scalar(0));
		cv::solve(X, Y, A, DecompTypes::DECOMP_QR | DecompTypes::DECOMP_NORMAL);

		//���������a��b��c��
		float _a, _b, _c, A0, A1, A2;
		A0 = A.at<float>(0, 0);
		A1 = A.at<float>(1, 0);
		A2 = A.at<float>(2, 0);

		_a = std::exp(A0 - (A1 * A1) / (4 * A2));
		_b = -1 / A2;
		_c = -A1 / (2 * A2);
		if (_b <= 0 || _a <= 0)
			continue;

		double sigma = sqrt(_b / 2);
		//�������Ч������
		/*float testvalue[5];
		for (int i = 0; i < 5; i++) {
			float x = interData.projection[i];
			testvalue[i] = a * (exp(-(x - c) * (x - c) / b));
		}*/

		float subPixelShift = _c - sigma;
		if (abs(subPixelShift) > 2)
			continue;

		//��������������  x = X+subPixelShift * cos(sinta),  y = Y+subPixelShift * sin(sinta)
		float sinta = angle360 / 180 * CV_PI;
		float sub_x = sample[0].x + subPixelShift * cos(sinta);
		float sub_y = sample[0].y + subPixelShift * sin(sinta);
		subPixelData.emplace_back(sub_x, sub_y);
		/*if (i == 60)
			cout << "pause";*/	
	}
 return;
}

#define KERNEL_SUM 8

void subPixelBasePolynomy(const cv::Mat& src, std::vector<cv::Point2f>& subPixelData) {
	
	Mat srcGray;
	if (src.channels() != 1)
		cvtColor(src, srcGray, COLOR_BGR2GRAY);
	else
		srcGray = src;

	Mat gauss;
	GaussianBlur(srcGray, gauss, Size(3, 3), 0);

	Matx33f kernels[KERNEL_SUM];
	int k = 0;
	kernels[k++] = { 1,2,1,0,0,0,-1,-2,-1 };//270��
	kernels[k++] = { 2,1,0,1,0,-1,0,-1,-2 };//315
	kernels[k++] = { 1,0,-1,2,0,-2,1,0,-1 };//0
	kernels[k++] = { 0,-1,-2,1,0,-1,2,1,0 };//45
	flip(kernels[0], kernels[k++], 0);//90
	flip(kernels[1], kernels[k++], -1);//135
	flip(kernels[2], kernels[k++], -1);//180
	flip(kernels[3], kernels[k++], -1);//225

	//�ݶ�ͼ��
	Mat gradients[KERNEL_SUM];
	
	for (k = 0; k < KERNEL_SUM; k++) {
		filter2D(gauss, gradients[k], CV_16S, kernels[k]);
	}

//�������ݶȷ�ֵ�뷽��
	//�Ƕ��б�
	const short angle_list[] = { 270,315,0,45,90,135,180,225 };

	//�ܷ�ֵ����
	Mat amplitude(gauss.size(), CV_16SC1, Scalar::all(0));
	//�ǶȾ��� �����ʼ���� -64��ֻ��Ϊ�˹�һ��֮������ʾ�Ƕ� 0��
	Mat angle(gauss.size(), CV_16SC1, Scalar::all(-64));

	for (int r = 0; r < gauss.rows; r++) {
		short* pAmp = amplitude.ptr<short>(r);
		short* pAng = angle.ptr<short>(r);
		short* pGrad[KERNEL_SUM] = { nullptr };

		for (int i = 0; i < KERNEL_SUM; i++) {
			pGrad[i] = gradients[i].ptr<short>(r);
		}

		for (int c = 0; c < gauss.cols; c++) {
			for (int i = 0; i < KERNEL_SUM; i++) {
				if (pAmp[c] < pGrad[i][c]) {
					pAmp[c] = pGrad[i][c];
					pAng[c] = angle_list[i];
				}
			}
		}

	}

//��Եϸ��
	double thresh = 128;
	Mat edge(gauss.size(), CV_8UC1, Scalar::all(0));

	for (int r = 1; r < gauss.rows - 1; r++) {
		//3*3�����ڣ�������3��ָ�룬һ��ָ��ָ��һ��
		const short* pAmp1 = amplitude.ptr<short>(r - 1);
		const short* pAmp2 = amplitude.ptr<short>(r);
		const short* pAmp3 = amplitude.ptr<short>(r + 1);

		const short* pAng = angle.ptr<short>(r);

		uchar* pEdge = edge.ptr<uchar>(r);

		for (int c = 1; c < gauss.cols - 1; c++) {
			//��ֵ�ж�
			if (pAmp2[c] < thresh)
				continue;
			//������������ȷ��������£�ʹ��switch����if�ж�
			switch (pAng[c]) {
			case 270:
				if (pAmp2[c] > pAmp1[c] && pAmp2[c] >= pAmp3[c])
					pEdge[c] = 255;
				break;
			case 90:
				if (pAmp2[c] >= pAmp1[c] && pAmp2[c] > pAmp3[c])
					pEdge[c] = 255;
				break;
			case 315:
				if (pAmp2[c] > pAmp1[c - 1] && pAmp2[c] >= pAmp3[c + 1])
					pEdge[c] = 255;
				break;
			case 135:
				if (pAmp2[c] >= pAmp1[c - 1] && pAmp2[c] > pAmp3[c + 1])
					pEdge[c] = 255;
				break;
			case 0:
				if (pAmp2[c] > pAmp2[c - 1] && pAmp2[c] >= pAmp2[c + 1])
					pEdge[c] = 255;
				break;
			case 180:
				if (pAmp2[c] >= pAmp2[c - 1] && pAmp2[c] > pAmp2[c + 1])
					pEdge[c] = 255;
				break;
			case 45:
				if (pAmp2[c] > pAmp3[c - 1] && pAmp2[c] >= pAmp1[c + 1])
					pEdge[c] = 255;
				break;
			case 225:
				if (pAmp2[c] >= pAmp3[c - 1] && pAmp2[c] > pAmp1[c + 1])
					pEdge[c] = 255;
				break;
			default:
				break;
			}
		}
	}
//���������ر�Ե��
	const double root2 = sqrt(2.0);
	//���Ǻ�����
	double tri_list[2][KERNEL_SUM] = { 0 };

	for (int i = 0; i < KERNEL_SUM; i++) {
		tri_list[0][i] = cos(angle_list[i] * CV_PI / 180.0);
		//sinǰ��ĸ��ŷǳ��ؼ�����Ϊͼ���Y������ֱ������ϵ��y�����෴��
		tri_list[1][i] = -sin(angle_list[i] * CV_PI / 180.0);
	}

	//vector ��ʽ��¼����������
	vector<Point2f> vPts;
	//Mat ��ʽ��¼���������꣬ע��������˫ͨ��
	Mat coordinate(gauss.size(), CV_32FC2, Scalar::all(0));

	for (int r = 1; r < gauss.rows - 1; r++) {
		//3*3����������3��ָ�룬һ��ָ��ָ��һ��
		const short* pAmp1 = amplitude.ptr<short>(r - 1);
		const short* pAmp2 = amplitude.ptr<short>(r);
		const short* pAmp3 = amplitude.ptr<short>(r + 1);

		const short* pAng = angle.ptr<short>(r);
		const uchar* pEdge = edge.ptr<uchar>(r);

		float* pCoordinate = coordinate.ptr<float>(r);

		for (int c = 1; c < gauss.cols - 1; c++) {
			if (pEdge[c])
			{
				int nAngTmp = 0;
				double dTmp = 0;

				switch (pAng[c])
				{
				case 270:
					nAngTmp = 0;
					dTmp = ((double)pAmp1[c] - pAmp3[c]) / (pAmp1[c] + pAmp3[c] - 2 * pAmp2[c]) * 0.5;
					break;
				case 90:
					nAngTmp = 4;
					dTmp = -((double)pAmp1[c] - pAmp3[c]) / (pAmp1[c] + pAmp3[c] - 2 * pAmp2[c]) * 0.5;
					break;
				case 315:
					nAngTmp = 1;
					dTmp = ((double)pAmp1[c - 1] - pAmp3[c + 1]) / (pAmp1[c - 1] + pAmp3[c + 1] - 2 * pAmp2[c]) * root2 * 0.5;
					break;
				case 135:
					nAngTmp = 5;
					dTmp = -((double)pAmp1[c - 1] - pAmp3[c + 1]) / (pAmp1[c - 1] + pAmp3[c + 1] - 2 * pAmp2[c]) * root2 * 0.5;
					break;
				case 0:
					nAngTmp = 2;
					dTmp = ((double)pAmp2[c - 1] - pAmp2[c + 1]) / (pAmp2[c - 1] + pAmp2[c + 1] - 2 * pAmp2[c]) * 0.5;
					break;
				case 180:
					nAngTmp = 6;
					dTmp = -((double)pAmp2[c - 1] - pAmp2[c + 1]) / (pAmp2[c - 1] + pAmp2[c + 1] - 2 * pAmp2[c]) * 0.5;
					break;
				case 45:
					nAngTmp = 3;
					dTmp = ((double)pAmp3[c - 1] - pAmp1[c + 1]) / (pAmp3[c - 1] + pAmp1[c + 1] - 2 * pAmp2[c]) * root2 * 0.5;
					break;
				case 225:
					nAngTmp = 7;
					dTmp = -((double)pAmp3[c - 1] - pAmp1[c + 1]) / (pAmp3[c - 1] + pAmp1[c + 1] - 2 * pAmp2[c]) * root2 * 0.5;
					break;
				default:
					break;
				}
				const double x = c + dTmp * tri_list[0][nAngTmp];
				const double y = r + dTmp * tri_list[1][nAngTmp];

				//vector��ʽ
				subPixelData.emplace_back((float)x, (float)y);
				//Mat ��ʽ
				/*pCoordinate[c << 1] = (float)x;
				pCoordinate[c << 1 + 1] = (float)y;*/				
			}
		}	
	}
	return;
}

//�洢��Ե������ز���
typedef struct _tagEdgeParm
{
	int thres;
	int parts;
}EDGE_PARAM;

#define KERNEL_HALF 4
// imgsrc: ���ͼ��, CV_8UC1
// edge: ���������Եͼ��
// vPts: �����¼ vector
// thres: �ݶ���ֵ
// parts: ͼ��ֿ���, ��ͼ��Ƚ�Сʱ, ��û�зֿ�ı�Ҫ��
void subPixelPolyFaster(Mat& imgsrc, Mat& edge, vector<Point2f>& vPts, int thres, int parts)
{
	 Matx33f kernels[KERNEL_SUM];//�˲���ģ��

	int k = 0;
	kernels[k++] = { 1,2,1,0,0,0,-1,-2,-1 };//270��
	kernels[k++] = { 2,1,0,1,0,-1,0,-1,-2 };//315
	kernels[k++] = { 1,0,-1,2,0,-2,1,0,-1 };//0
	kernels[k++] = { 0,-1,-2,1,0,-1,2,1,0 };//45

	//flip(kernels[0], kernels[k++], 0);//90
	//flip(kernels[1], kernels[k++], -1);//135
	//flip(kernels[2], kernels[k++], -1);//180
	//flip(kernels[3], kernels[k++], -1);//225
	
	// �ݶ�ͼ������
	Mat gradients[KERNEL_SUM];

	EDGE_PARAM edge_param;
	
	thread thread1[KERNEL_HALF];
	//ʹ�ö��̼߳�lamda���ʽ��ͼ������˲���ÿ���˲��˷���һ���̡߳�
	//��������ʹ�ö��߳�ʱ������ʹ����ڴ棬��ʡ����������ʱ��
	for (int i = 0; i < KERNEL_HALF; i++)
	{
		thread1[i]=std::thread ([](Mat* src, Mat *grad, Matx33f *ker, EDGE_PARAM* param)//lamda���ʽ//�̲߳��������ñ���ʹ��std::ref��std::cref;
			{
				filter2D(*src, *grad, CV_16S, *ker);//���
				*(grad + KERNEL_HALF) = -(*grad);//�Լ��Ϊ4�ľ���ֵȡ������Ϊ�˲��˵ĶԳ��ԡ�
			},&imgsrc,&gradients[i],&kernels[i],&edge_param);
		
	}
	//�����߳�
	for (int id = 0; id < KERNEL_HALF; id++)
	{
		thread1[id].join();
	}
	//��ֵ�ͽǶȾ���ϲ���һ������������CV_16SC2
	//�´�����ͼ�����������ģ����Կ��԰������������Ч��
	Mat amp_ang(imgsrc.rows, imgsrc.cols, CV_16SC2, Scalar::all(0));

	edge_param.parts = parts;//�ֿ�����
	assert(parts >= 1 && parts < (amp_ang.rows >> 1));

	vector<std::thread> thread2(edge_param.parts);

	for (int i = 0; i < parts; i++) {
		thread2[i] = std::thread ([](Mat* amp_ang, Mat* grad, int cur_part, EDGE_PARAM* param)
			{
				const int length = amp_ang->rows * amp_ang->cols;//�����ڴ���ܳ���,1 * length��
				const int step = length / param->parts; //�ֿ�ʱ��ÿһ��Ӧ�ö���ڴ档

				const int start = cur_part * step;//ÿ�ηֿ����ʼλ�ã�������
				int end = start + step;//ÿ�ηֿ����ֹλ�ã���������[start,ebd����

				if (cur_part >= param->parts - 1)//�����ʣ��Ĳ���ȫ����ᵽ���һ�ηֿ顣
				{
					end = length;
				}
				//ָ������ڴ�����ݵ�ַ��data��ʾ�ڴ���ʼλ�ã�start<<1����start*2����ʾ��ַƫ��������Ϊ��˫ͨ����
				//����������λ���������2��ָ�������ĳ˷����������������͡�
				short* amp_ang_ptr = (short*)amp_ang->data + (start << 1);
				
				short* grad_ptr[KERNEL_SUM] = { nullptr };//��8��ģ���е��ݶ�����λ����ָ������������������ҳ����ֵ��
				
				//ȷ�������ݶ�ͼ����������
				for (int ker_id = 0; ker_id < KERNEL_SUM; ker_id++)
				{
					
					assert((*(grad + ker_id)).isContinuous());
				}

				//��ȡ�ݶ�ͼ����ͬһλ�õ��ݶȣ�������洢��
				for (int k = 0; k < KERNEL_SUM; k++)
				{
					grad_ptr[k] = (short*)grad[k].data + start;//�洢��ַshort* ���͵�ָ�롣
				}

				for (int j = start; j < end; j++)
				{
					//�ҳ����ֵ���жϷ���
					for (int k = 0; k < KERNEL_SUM; k++)
					{
						if (*amp_ang_ptr < *grad_ptr[k])
						{
							*amp_ang_ptr = *grad_ptr[k];//ֱ��ȡ��ַ�Ƚ��븳ֵ
							*(amp_ang_ptr + 1) = k;
						}
						grad_ptr[k]++;//��ַ��1��ָ����һ������,���뱣֤�ݶ�ͼ����ڴ��������ġ�
					}
					amp_ang_ptr += 2;//����ڴ��2����Ϊ��˫ͨ��
				}							
		},&amp_ang, gradients,i,&edge_param);
		
	}
	//�����߳�
	for (int id = 0; id < edge_param.parts; id++)
	{
		thread2[id].join();
	}

	edge_param.thres = thres;
	edge = Mat::zeros(amp_ang.rows, amp_ang.cols,CV_8UC1);
	//������������ÿ���߳�ʹ�ò�ͬ�Ĵ洢��������󽫲�ͬ�����ڵ����ݲ��뵽һ�������У�������ʹ���ռ䣬
	vector<vector<Point2f>> vvtmp(parts);

	vector<std::thread> thread3(edge_param.parts);
	for (int i = 0; i < parts; i++)
	{
		thread3[i] =std::thread ([](Mat* amp_ang, Mat* edge, vector<Point2f>* v, int cur_part, EDGE_PARAM* param)
			{
				//�����������߳�֮���ֻ����������ʹ��static��ʼ��һ�μ��ɣ�
				//�����������ⶨ�����ⲿ�����̴߳��������鷳���Ͼ�ֻ���ⲿ��ʹ�á�
				static const float root2 = (float)sqrt(2.0);
				static const float angle2rad = (float)(CV_PI / 180.0);
				static const short angle_list[] = { 270,315,0,45,90,135,180,225 };

				//���Ǻ�����
				//�������������ڱ���μ���Ķ�ֵ�����Ը��ݲ����߼���Ч�ʣ��ر��Ƕ������Ǻ�����������ָ��Ӻ���
				float tri_list[2][KERNEL_SUM] = { 0 };
				
				for (int j = 0; j < KERNEL_SUM; j++) {
					if (angle_list[j] % 90 == 0) 
					{
						tri_list[0][j] = (float)(0.5f * cos(angle_list[j] * angle2rad));
						//0.5ǰ��ĸ��ŷǳ��ؼ�����Ϊͼ���y�����ֱ������ϵ��y�����෴
						tri_list[1][j] = (float)(-0.5f * sin(angle_list[j] * angle2rad));
					}
					else 
					{						
						tri_list[0][j] = (float)(0.5f * cos(angle_list[j] * angle2rad)) * root2;
						tri_list[1][j] = (float)(-0.5f * sin(angle_list[j] * angle2rad)) * root2;
					}				
				}

				const int thres = param->thres;
				const int rows_step = (amp_ang->rows -2) / param->parts;//���������µ����в����㣬������������ͬ����

				int start_y = rows_step * cur_part + 1;//������һ��
				int end_y = start_y + rows_step;//��Χ[start_y,end_y)
	
				if (cur_part >= param->parts - 1)//��������ʣ����в������һ��
				{
					end_y = amp_ang->rows-1;//���һ�в�Ҫ��
				}

				v->reserve(((end_y - start_y)>> 2) * (amp_ang->cols-2));
			
			
				for (int r = start_y; r < end_y; r++)
				{
					//3* 3����������3��ָ�룬һ��ָ��ָ��һ��
					const short* pAmpang1 = amp_ang->ptr<short>(r - 1);
					const short* pAmpang2 = amp_ang->ptr<short>(r);
					const short* pAmpang3 = amp_ang->ptr<short>(r + 1);

					uchar* pEdge = edge->ptr<uchar>(r);

					for (int c = 1; c < amp_ang->cols-1; c++)
					{
						const int j = c << 1;//���䶨����Χ���ǲ���ģ�������const;
						if (pAmpang2[j] >= thres)
						{					
							switch (pAmpang2[j + 1])
							{								
							case 0:
								if (pAmpang2[j] > pAmpang1[j] && pAmpang2[j] >= pAmpang3[j])
								{
									pEdge[c] = 255;

									v->emplace_back((float)c, r + tri_list[1][pAmpang2[j + 1]] * (pAmpang1[j] - pAmpang3[j]) 
										/ (pAmpang1[j] + pAmpang3[j] - (pAmpang2[j] << 1)));

								}
								break;
							case 4:
								if (pAmpang2[j] > pAmpang3[j] && pAmpang2[j] >= pAmpang1[j])
								{
									pEdge[c] = 255;

									v->emplace_back((float)c, r - tri_list[1][pAmpang2[j + 1]] *(pAmpang1[j] - pAmpang3[j]) 
										/(pAmpang1[j] + pAmpang3[j] - (pAmpang2[j] << 1)));
								}
								break;
							case 1:
								if (pAmpang2[j] > pAmpang1[j - 2] && pAmpang2[j] >= pAmpang3[j + 2])
								{
									pEdge[c] = 255;

									const float tmp = (float)(pAmpang1[j - 2] - pAmpang3[j + 2]) 
										/ (pAmpang1[j - 2] + pAmpang3[j + 2] - (pAmpang2[j] << 1));
									v->emplace_back(c + tmp * tri_list[0][pAmpang2[j + 1]], 
												   r + tmp * tri_list[1][pAmpang2[j + 1]]);
								}
								break;
							case 5:
								if (pAmpang2[j] > pAmpang3[j + 2] && pAmpang2[j] >= pAmpang1[j - 2])
								{
									pEdge[c] = 255;

									const float tmp = (float)(pAmpang1[j - 2] - pAmpang3[j + 2]) /
										(pAmpang1[j - 2] + pAmpang3[j + 2] - (pAmpang2[j] << 1));

									v->emplace_back(c - tmp * tri_list[0][pAmpang2[j + 1]],
												   r - tmp * tri_list[1][pAmpang2[j + 1]]);
								}
								break;
							case 2:
								if (pAmpang2[j] > pAmpang2[j - 2] && pAmpang2[j] >= pAmpang2[j + 2])
								{
									pEdge[c] = 255;

									v->emplace_back(c + tri_list[0][pAmpang2[j + 1]] * (pAmpang2[j - 2] - pAmpang2[j + 2]) /
										(pAmpang2[j - 2] + pAmpang2[j + 2] - (pAmpang2[j] << 1)), (float)r);
								}
								break;
							case 6:
								if (pAmpang2[j] > pAmpang2[j + 2] && pAmpang2[j] >= pAmpang2[j - 2])
								{
									pEdge[c] = 255;
								
									v->emplace_back(c - tri_list[0][pAmpang2[j + 1]] * (pAmpang2[j - 2] - pAmpang2[j + 2]) /
										(pAmpang2[j - 2] + pAmpang2[j + 2] - (pAmpang2[j] << 1)),(float)r);
								}
								break;
							case 3:
								if (pAmpang2[j] > pAmpang3[j - 2] && pAmpang2[j] >= pAmpang1[j + 2])
								{
									pEdge[c] = 255;

									const float tmp = (float)(pAmpang3[j - 2] - pAmpang1[j + 2]) /
										(pAmpang1[j + 2] + pAmpang3[j - 2] - (pAmpang2[j] << 1));

									v->emplace_back(c + tmp * tri_list[0][pAmpang2[j + 1]],
										r + tmp * tri_list[1][pAmpang2[j + 1]]);

								}
								break;
							case 7:
								if (pAmpang2[j] > pAmpang1[j + 2] && pAmpang2[j] >= pAmpang3[j - 2])
								{
									pEdge[c] = 255;

									const float tmp = (float)(pAmpang3[j - 2] - pAmpang1[j + 2]) /
										(pAmpang1[j + 2] + pAmpang3[j - 2] - (pAmpang2[j] << 1));

									v->emplace_back(c - tmp * tri_list[0][pAmpang2[j + 1]],
										r - tmp * tri_list[1][pAmpang2[j + 1]]);
								}
								break;
							default:
								break;
							}						
						}
					}
				}		
			},&amp_ang,&edge,&vvtmp[i],i, &edge_param);			
	}
	//�����߳�
	for (int id = 0; id < edge_param.parts; id++)
	{
		thread3[id].join();
	}

	for (int i = 0; i < parts; i++)
	{ 
		//�������̵߳�������ݺϲ���ͬһ�������С�
		vPts.insert(vPts.end(), vvtmp[i].begin(), vvtmp[i].end());
	}
}