#include<opencv2/opencv.hpp>
using namespace cv;
#define MY_FAIL -1
#define MY_DONE 1
#define MY_PI 3.14159f
//#define IMG_SHOW
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg); //转灰度
int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th); //二值化
int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y);//sobel算子
int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat&angleImg, Mat& magImg); //计算角度和幅值
int ustc_CalcHist(Mat grayImg, int* hist, int hist_len);//灰度直方图
int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y); //亮度比较 子块匹配
int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y); //色彩比较 子块匹配
int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y); //基于角度比较的子块匹配
int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y); //基于幅值比较的子块匹配
int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y);//基于直方图比较的子块匹配
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y);//基于相关性的子块匹配
inline float myatan2(float y, float x);//反正切
