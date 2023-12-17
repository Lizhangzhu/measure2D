#pragma once
#include<opencv2/opencv.hpp>
#include<vector>




void subPixelBaseGauss(const cv::Mat& src, std::vector<cv::Point2f>& subPixelData,const double thres);

void subPixelBasePolynomy(const cv::Mat& src, std::vector<cv::Point2f>& subPixelData);

void subPixelPolyFaster(cv::Mat& imgsrc, cv::Mat& edge, std::vector<cv::Point2f>& vPts, int thres, int parts);

