//
// author: Jerry Li
//
#ifndef IMAGEPROCESSINGFROM_RANSAC_H
#define IMAGEPROCESSINGFROM_RANSAC_H
#include <opencv2/opencv.hpp>
#include <vector>
#include <mutex>
#include "windows.h"
#define PI 3.14159265358979323846 


//--------------------------------------------------------------------- Ransac ----------------------------------------------------------------------------------------------------------
template<class _Tp>
class Ransac {
#define THISCLASS Ransac
public: 
	Ransac(const std::vector<_Tp>& data, double confidence, double EPS, double innerPercent,int modelSize, int modelNumberOfPoints, bool olsFlag = false,int threadNum = -1) :
		m_data(data), m_confidence(confidence), m_EPS(EPS), m_innerPercent(innerPercent), m_olsFlag(olsFlag),m_threadNum(threadNum),m_model(modelSize,0),m_modelNumberOfPoints(modelNumberOfPoints) {
	}
	virtual ~Ransac() = 0 {}//虚析构使得该类成为抽象类，虚函数必须有函数体，子类析构时会调用父类的析构函数。
	
	//得到模型参数
	std::vector<double> getModelParam() const {
		return this->m_model;
	}

	//得到内点集
	std::vector<_Tp> getInner() const {
		return this->m_inner;
	}

	void setConfidence(double value) {
		assert(value > 0);
		this->m_confidence = value;
	}
	
	void  setEPS(double value) {
		assert(value > 0);
		this->m_EPS = value;
	}
	
	void setInnerPercent(double value) {
		assert(value > 0);
		this->m_confidence = value;
	}

	void setOlsFlag(bool value) {
		this->m_olsFlag = value;
	}

	void setThreadNum(int value) {
		assert(value != 0);
		m_threadNum = value;
	}

	void fitModel();//拟合流程主函数

protected:
	//--------------------------------member variable -------------------------------------------
		//--------------------------------------input-------------------------------------
	std::vector<_Tp> m_data;//样本点数据

	int m_modelNumberOfPoints;//每次抽取样本点的数量

	int m_modelSize;//模型参数的数量
	
	double m_confidence;//置信度

	double m_EPS;//度量阈值

	double m_innerPercent;//预估内点占比

	bool m_olsFlag;//是否使用最小二乘法

	int m_threadNum;//线程数量

		//--------------------------------------output------------------------------------
	std::vector<double> m_model;//最终模型

	std::vector<_Tp> m_inner;//最终模型拥有的内点集；

	
	//------------------------------------member function---------------------------------
	//计算迭代次数
	int calcIterateTimes(int sampleNumber) {
		unsigned long iterateTimes = (unsigned)ceil(log(1 - this->m_confidence) / log(1 - pow(this->m_innerPercent, sampleNumber)));
		return iterateTimes;
	}

	//计算线程数量
	virtual  int calcNumberOfThread(int iterateTimes) = 0;

	//计算内点数量不保留内点
	virtual int calcInner(const std::vector<double>& model) = 0;

	//计算内点数量只保留内点
	virtual void calcInner(const  std::vector<double>& model, std::vector<_Tp>& inner) = 0;

	//最小二乘法拟合
	virtual void ols(std::vector<double>& m_model, const std::vector<_Tp>& m_inner) = 0;
	
	//线程回调函数
	virtual void calcInnerThread(const std::vector<_Tp>& data, std::vector<double>& model, int& number, int iterateTimes) = 0;
	
};

template<class _Tp>
void Ransac<_Tp>::fitModel() {
	//清空上一次拟合的容器内存
	this->m_inner.clear();
	this->m_inner.reserve(this->m_data.size() * this->m_innerPercent);//预留内存减少动态扩容的次数

	//确定迭代次数k
	int iterateTimes = this->calcIterateTimes(this->m_modelNumberOfPoints);
	//std::cout << "迭代次数k: " << k<<std::endl;

	//设置随机数种子
	srand(unsigned(time(NULL)));

	//如果调用者指定了线程数量，则使用指定参数。
	int threadNum;
	if (this->m_threadNum > 0 ) 
		threadNum = m_threadNum;
	else 
		threadNum = this->calcNumberOfThread(iterateTimes);
	
	//计算每个线程的迭代次数，尽可能均匀分布。
	int iterateEveryTime = iterateTimes / threadNum;
	int remainder = iterateTimes % threadNum;//remainder个数的线程的迭代次数需要在iterateEveryTime加1
		
	//创建线程参数;
	std::vector<std::vector<double>> thread_model(threadNum, std::vector<double>(m_modelSize, 0));
	std::vector<int> thread_inner(threadNum, 0);

	//2.划分迭代次数创建线程
	std::vector<std::thread> threadArray(threadNum);
	for (unsigned int threadId = 0; threadId < remainder; threadId++) {
		//创建线程，每个线程输出其迭代次数内最好的模型和内点数
		threadArray[threadId] = std::thread(&THISCLASS::calcInnerThread, this, std::cref(this->m_data), 
			std::ref(thread_model[threadId]), std::ref(thread_inner[threadId]), iterateEveryTime+1);
	}
	for (unsigned int threadId = remainder; threadId < threadNum; threadId++) {
		threadArray[threadId] = std::thread(&THISCLASS::calcInnerThread, this, std::cref(this->m_data),
			std::ref(thread_model[threadId]), std::ref(thread_inner[threadId]), iterateEveryTime);
	}

	//回收线程资源
	for (unsigned int threadId = 0; threadId < threadNum; threadId++) {
		threadArray[threadId].join();
	}
	//在输出模型中寻找内点数量最多的模型;
	int best_inner = 0;
	for (unsigned int i = 0; i < threadNum; i++) {
		if (thread_inner[i] > best_inner) {
			best_inner = thread_inner[i];
			this->m_model = thread_model[i];
		}
	}
	
	//计算内点，并将内点加载进m_inner;
	calcInner(this->m_model, this->m_inner);
// --------------------------------测试代码---------------------------------------

	//最小二乘法优化
	if (this->m_olsFlag) {
		ols(this->m_model, this->m_inner);
	}
	return;
}

//--------------------------------------------------------------------- RansacLine ----------------------------------------------------------------------------------------------------------
template<class _Tp>
class RansacLine:public Ransac<_Tp> {
//typedef RansacLine THISCLASS;
#define THISCLASS RansacLine //定义此类型便于类内线程调用成员函数；
private:	
	//计算创建线程数量
	int calcNumberOfThread(int iterateTimes);

	//计算内点数量不保留内点
	int calcInner(const std::vector<double>& model);

	//计算内点数量且保留内点
	void calcInner(const  std::vector<double>& model, std::vector<_Tp>& inner); 

	//线程回调函数
	void calcInnerThread(const std::vector<_Tp>& data, std::vector<double>& model, int& number, int iterateTimes); 

	//最小二乘法拟合
	void ols(std::vector<double>& m_model, const std::vector<_Tp>& m_inner);
public:
	RansacLine(const std::vector<_Tp>& data, double confidence, double EPS, double innerPercent, bool olsFlag = false) 
		:Ransac<_Tp>( data, confidence,  EPS,  innerPercent,3,2,olsFlag )
	{
		
	}
	~RansacLine(){}

};

template<class _Tp>
int RansacLine<_Tp>::calcNumberOfThread(int iterateTimes) {
	//1.根据数据量与本机CPU支持的核心数确定线程数量;
	SYSTEM_INFO sysInfo;
	GetSystemInfo(&sysInfo);
	//目前电脑6核，超线程技术模拟成12个逻辑处理器。
	int threadNum = 0;
	unsigned int  dataSize = this->m_data.size();
	if (dataSize < 5120) {
		threadNum = 1;
	}
	else if (dataSize < 10240) {
		threadNum = sysInfo.dwNumberOfProcessors / 6;
	}
	else if (dataSize < 40960) {
		threadNum = sysInfo.dwNumberOfProcessors / 3;
	}
	else if (dataSize < 163804) {
		threadNum = sysInfo.dwNumberOfProcessors / 2;
	}
	else {
		threadNum = sysInfo.dwNumberOfProcessors;
	}
	return threadNum;
}

template<class _Tp>
void RansacLine<_Tp>::calcInner(const  std::vector<double>& model, std::vector<_Tp>& inner) {
	//计算最佳模型的内点
	
	double distDenominator = sqrt(pow(model[0], 2) + pow(model[1], 2));
	double EPSTotal = this->m_EPS * distDenominator;
	unsigned int data_size = this->m_data.size();

	for (unsigned long i = 0; i < data_size; i++) {
		//点到直线距离公式
		double distance = abs(model[0] * this->m_data[i].x + model[1] * this->m_data[i].y + model[2]);
		if (distance < EPSTotal) {
			inner.push_back(this->m_data[i]);
		}
	}
	return;
}

template<class _Tp>
int  RansacLine<_Tp>::calcInner(const std::vector<double>& model) {
	//计算最佳模型的内点
	int  inner_number = 0;
	double distDenominator = sqrt(pow(model[0], 2) + pow(model[1], 2));
	double EPSTotal = this->m_EPS * distDenominator;
	unsigned int data_size = this->m_data.size();

	for (unsigned long i = 0; i < data_size; i++) {
		//点到直线距离公式
		double distance = abs(model[0] * this->m_data[i].x + model[1] * this->m_data[i].y + model[2]);
		if (distance < EPSTotal) {
			inner_number++;
		}
	}
	return inner_number;
}

template<class _Tp>
 void RansacLine<_Tp>:: ols(std::vector<double>& m_model, const std::vector<_Tp>& m_inner) {
	//最小二乘法在近乎垂直的时候不适用,而总体最小二乘法对于求解非常复杂。
	//对于小角度采用与X或者Y轴的距离直接做最小二乘法，对于大角度将内点集旋转到
	//某一坐标轴附近进行最小二乘，得到直线方程后再旋转回来
	//1.近似垂直的情况，X作为误差观测值 X= kY+b;
	if (this->m_model[1] == 0 || abs(atan(-this->m_model[0] / this->m_model[1]) * 180 / PI) > 80) {
		double y_ = 0, x_ = 0, y_2 = 0, y_x_ = 0;
		for (unsigned int id = 0; id < this->m_inner.size(); id++) {
			y_ += this->m_inner[id].y;
			x_ += this->m_inner[id].x;
			y_x_ += int(this->m_inner[id].y * (double)this->m_inner[id].x);
			y_2 += int(this->m_inner[id].y * (double)this->m_inner[id].y);
		}

		double denominator = this->m_inner.size() * y_2 - y_ * y_;// 此处已保证不为0
		double line_k = (this->m_inner.size() * y_x_ - y_ * x_) / denominator;
		double line_b = (y_2 * x_ - y_ * y_x_) / denominator;
		//点斜式转换成一般式
		this->m_model[0] = -1;
		this->m_model[1] = line_k;
		this->m_model[2] = line_b;

		//std::cout << line_k << " " << line_b <<std::endl;
	}
	//2.近似水平的情况，Y作为误差观测值 Y = kX+b;
	else if (abs(atan(-this->m_model[0] / this->m_model[1]) * 180 / PI) < 10) {
		double x_ = 0, y_ = 0, x_2 = 0, x_y_ = 0;
		for (unsigned int id = 0; id < this->m_inner.size(); id++) {
			y_ += this->m_inner[id].y;
			x_ += this->m_inner[id].x;
			x_y_ += int(this->m_inner[id].y * (double)this->m_inner[id].x);
			x_2 += int(this->m_inner[id].x * (double)this->m_inner[id].x);
		}
		double denominator = this->m_inner.size() * x_2 - x_ * x_;
		double line_k = (this->m_inner.size() * x_y_ - y_ * x_) / denominator;
		double line_b = (x_2 * x_ - y_ * x_y_) / denominator;

		this->m_model[0] = line_k;
		this->m_model[1] = -1;
		this->m_model[2] = line_b;
	}
	//3.正常角度使用旋转到X轴再计算。
	else {
		double theta = atan(-this->m_model[0] / this->m_model[1]);
		std::vector<_Tp> trans_data;
		//旋转-theta角度。
		float arr[] = { cos(theta),sin(theta),-sin(theta),cos(theta) };
		cv::Mat M(cv::Size(2, 2), CV_32FC1, arr);
		//数据点旋转
		cv::transform(this->m_inner, trans_data, M);
		//使用Y值作为观测误差 Y = kX + b;
		double x_ = 0, y_ = 0, x_2 = 0, x_y_ = 0;
		unsigned int inner_size = this->m_inner.size();
		for (unsigned int id = 0; id < inner_size; id++) {
			y_ += trans_data[id].y;
			x_ += trans_data[id].x;
			x_y_ += int(trans_data[id].y * (double)trans_data[id].x);
			x_2 += int(trans_data[id].x * (double)trans_data[id].x);
		}
		double denominator = trans_data.size() * x_2 - x_ * x_;
		double line_k = (trans_data.size() * x_y_ - y_ * x_) / denominator;
		double line_b = (x_2 * y_ - x_ * x_y_) / denominator;
		//旋转矩阵的逆
		this->m_model[0] = line_k * cos(theta) + sin(theta);
		this->m_model[1] = line_k * sin(theta) - cos(theta);
		this->m_model[2] = line_b;
		
	}
}

 template<class _Tp>
 void RansacLine<_Tp>::calcInnerThread(const std::vector<_Tp>& data, std::vector<double>& model, int& number, int iterateTimes) {
	 for (unsigned int it = 0; it < iterateTimes; it++) {
		 assert(iterateTimes > 1);
		 //抽样，保证两个点不重合
		 int index1, index2;
		 //若构造的类为模板类，那么派生类不可以直接使用继承到的基类数据和方法，
		 //需要通过this指针使用。否则，在使用一些较新的编译器时，会报“找不到标识符”错误。
		
		 index1 = (rand() % this->m_data.size());
		 do {
			 index2 = (rand() % this->m_data.size());
		 } while (index1 == index2);
		 cv::Point2i point_1 = this->m_data[index1];
		 cv::Point2i point_2 = this->m_data[index2];

		 //计算模型 (y1-y2)x + (x2-x1)y + x1y2 - x2y1 = 0
		 std::vector<double> cur_model(3);
		 cur_model[0] = (double)point_1.y - point_2.y;
		 cur_model[1] = (double)point_2.x - point_1.x;
		 cur_model[2] = (double)point_1.x * point_2.y - (double)point_2.x * point_1.y;

		 int inner_number = calcInner(cur_model);
		 //更新内点数和最佳模型；
		 if (inner_number > number) {
			 model = cur_model;
			 number = inner_number;
		 }
	 }
 }



 //--------------------------------------------------------------------- RansacCircle ----------------------------------------------------------------------------------------------------------
template<class _Tp>
class RansacCircle : public Ransac<_Tp> {
#define THISCLASS RansacCircle //定义此类型便于类内线程调用成员函数；

public:
	RansacCircle(const std::vector<_Tp>& data,double confidence,double EPS, double innerPercent,bool olsFlag)
		:Ransac<_Tp>(data, confidence, EPS, innerPercent,3, 3,olsFlag)
	{

	}

	~RansacCircle(){}

protected:
	//计算创建线程数量
	int calcNumberOfThread(int iterateTimes);

	//线程回调函数
	void calcInnerThread(const std::vector<_Tp>& data, std::vector<double>& model, int& number, int iterateTimes);

	//计算内点数量不保留内点
	int calcInner(const std::vector<double>& model);

	//计算内点数量只保留内点
	void calcInner(const  std::vector<double>& model, std::vector<_Tp>& inner);

	//最小二乘法拟合
	void ols(std::vector<double>& m_model, const std::vector<_Tp>& m_inner);
};

template<class _Tp>
int RansacCircle<_Tp>::calcNumberOfThread(int iterateTimes) {
	//1.根据数据量与本机CPU支持的核心数确定线程数量;
	SYSTEM_INFO sysInfo;
	GetSystemInfo(&sysInfo);
	//目前电脑6核，超线程技术模拟成12个逻辑处理器。
	int threadNum = 0;
	unsigned int  dataSize = this->m_data.size();
	if ( dataSize  < 113) {
		threadNum = 1;
	}
	else if ( dataSize < 306) {
		threadNum = sysInfo.dwNumberOfProcessors / 6;
	}
	else if (dataSize < 831) {
		threadNum = sysInfo.dwNumberOfProcessors / 3;
	}
	else if (dataSize < 2262) {
		threadNum = sysInfo.dwNumberOfProcessors / 2;
	}
	else {
		threadNum = sysInfo.dwNumberOfProcessors;
	}
	return threadNum;
}


template<class _Tp>
int RansacCircle<_Tp>::calcInner(const std::vector<double>& model) {
	//计算最佳模型的内点
	int  inner_number = 0;
	unsigned int data_size = this->m_data.size();

	for (unsigned long i = 0; i < data_size; i++) {
		//点到直线距离公式
		double distance = abs(sqrt(pow(this->m_data[i].x-model[0],2) + pow(this->m_data[i].y - model[1], 2))-model[2]);
		if (distance < this->m_EPS) {
			inner_number++;
		}
	}
	return inner_number;
}

template<class _Tp>
void RansacCircle<_Tp>::calcInner(const std::vector<double>& model,std::vector<_Tp>& inner ) {
	//计算最佳模型的内点
	unsigned int data_size = this->m_data.size();

	for (unsigned long i = 0; i < data_size; i++) {
		//点到直线距离公式
		double distance = abs(sqrt(pow(this->m_data[i].x - model[0], 2) + pow(this->m_data[i].y - model[1], 2)) - model[2]);
		if (distance < this->m_EPS) {
			inner.push_back(this->m_data[i]);
		}
	}
	return ;
}

template<class _Tp>
void RansacCircle<_Tp>::ols(std::vector<double>& m_model, const std::vector<_Tp>& m_inner) {

	//std::cout << "Ransac   X:" << m_model[0] << " Y:" << m_model[1] << " R:" << m_model[2] << std:: endl;
	/*cv::Mat show = test.clone();
	cv::circle(show, cv::Point(m_model[0], m_model[1]), m_model[2], cv::Scalar(0,0,255),1 );*/

	double X1 = 0, X2 = 0, X3 = 0, Y1 = 0, Y2 = 0, Y3 = 0, X1_Y1 = 0, X1_Y2 = 0, X2_Y1 = 0;
	for (const auto it : m_inner) {
		X1 += it.x;
		X2 += pow(it.x, 2);
		X3 += pow(it.x, 3);
		Y1 += it.y;
		Y2 += pow(it.y, 2);
		Y3 += pow(it.y, 3);
		X1_Y1 += it.x * it.y;
		X1_Y2 += it.x * pow(it.y, 2);
		X2_Y1 += pow(it.x, 2) * it.y;
 	}
	double C, D, E, G, H, N;
	double a, b, c;
	N = m_inner.size();
	C = N * X2 - X1 * X1;
	D = N * X1_Y1 - X1 * Y1;
	E = N * X3 + N * X1_Y2 - (X2 + Y2) * X1;
	G = N * Y2 - Y1 * Y1;
	H = N * (X2_Y1 + Y3) - (X2 + Y2) * Y1;

	a = (H * D - E * G) / (C * G - D * D);
	b = (H * C - E * D) / (D * D - G * C);
	c = -(a * X1 + b * Y1 + X2 + Y2) / N;


	m_model[0] = a / (-2);
	m_model[1] = b / (-2);
	m_model[2] = sqrt(a * a + b * b - 4 * c) / 2;
	//cv::Mat src = test.clone();
	/*std::cout << "ols   X:"<<m_model[0] << " Y:" << m_model[1] << " R:" << m_model[2] << std::endl;
	std::cout << std::endl;*/
	/*cv::circle(show, cv::Point(m_model[0], m_model[1]), m_model[2], cv::Scalar(0, 255, 0), 1);
	cv::waitKey(0);*/
}


template<class _Tp>//必须实现函数 1
void RansacCircle<_Tp>::calcInnerThread(const std::vector<_Tp>& data, std::vector<double>& model, int& number, int iterateTimes) {
	for (unsigned int it = 0; it < iterateTimes; it++) {
		assert(iterateTimes > 0);
		//抽样，保证三个点不重合不共线
		int id_1, id_2, id_3;
		//若构造的类为模板类，那么派生类不可以直接使用继承到的基类数据和方法，
		//需要通过this指针使用。否则，在使用一些较新的编译器时，会报“找不到标识符”错误。
		double value;
		do {
			id_1 = rand() % this->m_data.size();
			id_2 = rand() % this->m_data.size();
			id_3 = rand() % this->m_data.size();	//共线公式(y3−y1)(x2−x1)−(y2−y1)(x3−x1) = 0
			value = (data[id_3].y - data[id_1].y) * (data[id_2].x - data[id_1].x) - (data[id_2].y
				- data[id_1].y) * (data[id_3].x - data[id_1].x);
		} while (abs(value) < 0.000000001);
		//三点建立模型
		
		double a = data[id_1].x - data[id_2].x, b = data[id_1].y - data[id_2].y;
		double c = data[id_1].x - data[id_3].x, d = data[id_1].y - data[id_3].y;
		double e = ((data[id_1].x * data[id_1].x - data[id_2].x * data[id_2].x) - (data[id_2].y * data[id_2].y - data[id_1].y * data[id_1].y)) / 2;
		double f = ((data[id_1].x * data[id_1].x - data[id_3].x * data[id_3].x) - (data[id_3].y * data[id_3].y - data[id_1].y * data[id_1].y)) / 2;
		std::vector<double> cur_model(3);
		cur_model[0] = (d * e - b * f) / (a * d - b * c);
		cur_model[1] = (a * f - c * e) / (a * d - b * c);
		cur_model[2] = sqrt(pow(data[id_1].x - cur_model[0], 2) + pow(data[id_1].y - cur_model[1], 2));

		int inner_number = calcInner(cur_model);
		//更新内点数和最佳模型；
		if (inner_number > number) {
			model = cur_model;
			number = inner_number;
		}
	}
}
#endif//IMAGEPROCESSINGFROM_RANSAC_H