#include<opencv2/opencv.hpp>
#include<vector>
#include"subPixel.h"
#include <chrono>
#include "Spire.Xls.o.h"
#include "RANSAC.hpp"
using namespace std;
using namespace cv;
using namespace Spire::Xls;

#define GETDATA 

int main() {

#ifdef GETDATA
	//ָ������ļ�·��������
	std::wstring outputPath = L"F:\\algorithm\\";
	std::wstring outputFile = outputPath + L"subPixelDetectPolyFaseterCircle.xlsx";
	//����һ��Workbook����
	Workbook* workbook = new Workbook();
	//��ȡ��һ��������
	boost::intrusive_ptr<Spire::Xls::XlsWorksheet> sheet = workbook->GetWorksheets()->Get(0);
#endif // GETDATA
	
	Mat src = imread("������.jpg", IMREAD_COLOR);
	Mat srcGray;
	cvtColor(src, srcGray, COLOR_BGR2GRAY);
	vector<Point2f> subPixelData;
	
	Mat edge;
	auto start = chrono::steady_clock::now();
	int times = 10;
	for (int i = 0; i < times; i++)
	{
		//subPixelBasePolynomy(src, subPixelData);
		subPixelBaseGauss(srcGray, subPixelData,128);//6��������
		/*RansacCircle<Point2f> RansacCircle(subPixelData,0.999,1,0.5,true);
		RansacCircle.fitModel();
		vector<double> param = RansacCircle.getModelParam();
		circle(src, Point2i(cvFloor(param[0] + 0.5), cvFloor(param[1] + 0.5)), cvFloor(param[2]+0.5), Scalar(0, 255, 0), 1);*/
	}

#ifdef GETDATA
	for (int i = 0; i < subPixelData.size(); i++) {
		sheet->Get(i + 1, 1)->SetNumberValue(subPixelData[i].x);
		sheet->Get(i + 1, 2)->SetNumberValue(subPixelData[i].y);
	}
#endif // GETDATA

	auto end = chrono::steady_clock::now();
	auto durtime = end - start;
	cout << "��ʱ��" << durtime.count()  << "���루" << (double)durtime.count() / (1000.0 * 1000 * 1000) /times << "�룩" << endl;

#ifdef GETDATA
	workbook->SaveToFile(outputFile.c_str(), ExcelVersion::Version2010);
	workbook->Dispose();
#endif // GETDATA

	return 0;

	

}