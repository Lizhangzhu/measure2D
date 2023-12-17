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
	//指定输出文件路径和名称
	std::wstring outputPath = L"F:\\algorithm\\";
	std::wstring outputFile = outputPath + L"subPixelDetectPolyFaseterCircle.xlsx";
	//创建一个Workbook对象
	Workbook* workbook = new Workbook();
	//获取第一个工作表
	boost::intrusive_ptr<Spire::Xls::XlsWorksheet> sheet = workbook->GetWorksheets()->Get(0);
#endif // GETDATA
	
	Mat src = imread("极耳右.jpg", IMREAD_COLOR);
	Mat srcGray;
	cvtColor(src, srcGray, COLOR_BGR2GRAY);
	vector<Point2f> subPixelData;
	
	Mat edge;
	auto start = chrono::steady_clock::now();
	int times = 10;
	for (int i = 0; i < times; i++)
	{
		//subPixelBasePolynomy(src, subPixelData);
		subPixelBaseGauss(srcGray, subPixelData,128);//6毫秒左右
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
	cout << "耗时：" << durtime.count()  << "纳秒（" << (double)durtime.count() / (1000.0 * 1000 * 1000) /times << "秒）" << endl;

#ifdef GETDATA
	workbook->SaveToFile(outputFile.c_str(), ExcelVersion::Version2010);
	workbook->Dispose();
#endif // GETDATA

	return 0;

	

}