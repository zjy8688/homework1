#include  <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SubImageMatch.h"
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

//************  task1  ************//
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (bgrImg.data == NULL)
	{
		cout << "error  input is null" << endl;
		return(SUB_IMAGE_MATCH_FAIL);
	}
	int width = bgrImg.rows;
	int height = bgrImg.cols;
	int size = width*height; 
	grayImg.create(width, height, CV_8UC1);
	int i;
	int temp;
	for (i = 0; i < size; i++)
	{
		temp = 3 * i;
		grayImg.data[i] = (bgrImg.data[temp + 2]) * 0.299 + (bgrImg.data[temp + 1])* 0.587 + (bgrImg.data[temp + 0]) * 0.114;
	}
	return  SUB_IMAGE_MATCH_OK;

}
//************  task2  ************//
int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)                                //？？移位                //2.509ms
{

	if (grayImg.data == NULL)
	{
		cout << "error  input is null" << endl;
		return(SUB_IMAGE_MATCH_FAIL);
	}
	int width = grayImg.rows;
	int height = grayImg.cols;
	int size = width*height;
	int i,j;
	gradImg_x.create(width, height, CV_32FC1);
	gradImg_y.create(width, height, CV_32FC1);
	gradImg_x.zeros(width, height, CV_32FC1);
	gradImg_y.zeros(width, height, CV_32FC1);
	float* grad_x;
	float* grad_y;
	grad_x = (float*)gradImg_x.data;
	grad_y = (float*)gradImg_y.data;

	int temp;
	for (i = 1; i < width - 1; i++)
	{
		temp = i*height;
		for (j = 1; j < height - 1; j++)
		{
			//grad_x[temp + j] = (-1 * grayImg.data[temp-height + j - 1] + 1 * grayImg.data[temp-height + j + 1] - 2 * grayImg.data[temp+ j - 1]
			//	+ 2 * grayImg.data[temp + j + 1] - 1 * grayImg.data[temp+height + j - 1] + 1 * grayImg.data[temp+height + j + 1]);

			//grad_y[temp + j] = (-1 * grayImg.data[temp-height + j - 1] - 2 * grayImg.data[temp-height + j] - 1 * grayImg.data[temp - height + j + 1]
		    //	+ 1 * grayImg.data[temp + height + j - 1] + 2 * grayImg.data[temp + height + j] + grayImg.data[temp + height + j + 1]);

			grad_x[temp + j] = (- grayImg.data[temp - height + j - 1] +  grayImg.data[temp - height + j + 1] - ((grayImg.data[temp + j - 1])<<1)
				+  ((grayImg.data[temp + j + 1])<<1) -  grayImg.data[temp + height + j - 1] +  grayImg.data[temp + height + j + 1]);
			grad_y[temp + j] = (- grayImg.data[temp - height + j - 1] -  (grayImg.data[temp - height + j]<<1) - grayImg.data[temp - height + j + 1]
				+ grayImg.data[temp + height + j - 1] +  (grayImg.data[temp + height + j]<<1) + grayImg.data[temp + height + j + 1]);

		}
	}
	return(SUB_IMAGE_MATCH_OK);
}                    

//************  task3  ************//
int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)                        //角度图处理   角度范围
{
	if (gradImg_x.data == NULL|| gradImg_y.data == NULL)
	{
		cout << "error  input is null" << endl;
		return(SUB_IMAGE_MATCH_FAIL);
	}

	int width = gradImg_x.rows;
	int height = gradImg_y.cols;
	int size = width*height;
	angleImg.create(width, height, CV_32FC1);
	magImg.create(width, height, CV_32FC1);
	float* angle = (float*)angleImg.data;
	float* mag = (float*)magImg.data;
	float* grad_y = (float*)gradImg_y.data;
	float* grad_x = (float*)gradImg_x.data;

	int i;
	for (i = 0; i <size; i++)
	{
		if (gradImg_x.data[i] == 0)
		{
			angle[i] = 90;
			mag[i] = sqrt(grad_y[i] * grad_y[i] + grad_x[i] * grad_x[i]);
			continue;
		}
		angle[i] = 180*(atan2(grad_y[i] ,grad_x[i]))*1000/3141;
		if (angle[i] < 0)
			angle[i] = angle[i] + 180;
		mag[i] = sqrt(grad_y[i] * grad_y[i] + grad_x[i] * grad_x[i]);
	}
	return(SUB_IMAGE_MATCH_OK);
}

//************  task4  ************//
int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)                                      //0.67ms
{

	if (grayImg.data == NULL)
	{
		cout << "error  input is null" << endl;
		return(SUB_IMAGE_MATCH_FAIL);
	}
	int width = grayImg.rows;
	int height = grayImg.cols;
	int size = width*height;
	binaryImg.create(width, height, CV_8UC1);
	binaryImg.zeros(width, height, CV_8UC1);
	int i;
	for (i = 0; i < size; i++)
	{
		binaryImg.data[i] = ((th - grayImg.data[i]) >> 7 && 1) * 255;
	}

	return(SUB_IMAGE_MATCH_OK);
}

//************  task5  ************//
int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{

	if (grayImg.data == NULL)
	{
		cout << "error  input is null" << endl;
		return(SUB_IMAGE_MATCH_FAIL);
	}
	int width = grayImg.rows;
	int height = grayImg.cols;
	int size = width*height;
	for (int i = 0; i < size; i++)
	{
		hist[grayImg.data[i]] += 1;	
	}
	return(SUB_IMAGE_MATCH_OK);
}

//************  task6  ************//
int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{

	if (grayImg.data == NULL||subImg.data==NULL)
	{
		cout << "error  input is null" << endl;
		return(SUB_IMAGE_MATCH_FAIL);
	}
	int sub_width = subImg.rows;
	int sub_height = subImg.cols;
	int gray_width = grayImg.rows;
	int gray_height = grayImg.cols;


	if (sub_width > gray_width || sub_height > gray_height)
	{
		cout << "error the size of subimag oversizes the source one  ";
		return(SUB_IMAGE_MATCH_FAIL);
	}


	int sub_size = sub_width*sub_height;
	int gray_size = gray_width*gray_height;
    int diff[3] = { 0,0,0 };
	diff[0] = 255 * sub_size;
	int sum;
	int i, j, sub_i, sub_j;
	int temp;
	int subtemp, graytemp;

	
	/*for (i = 0; i < gray_width-sub_width; i++)
	{
		for (j = 0; j < gray_height-sub_height; j++)
		{
			sum = 0;
			for (sub_i = 0; sub_i <  sub_width; sub_i++)
			{
				subtemp = sub_i*sub_height;
				graytemp = (i + sub_i)*gray_height+j;
				for (sub_j = 0; sub_j <  sub_height; sub_j++)
				{
					temp = subImg.data[subtemp + sub_j] - grayImg.data[graytemp + sub_j];
					sum =sum+ ((temp >> 31) & 1)*(0 - 2 * temp) + temp;
				}
			}
			if (sum < diff[0])
			{
				diff[0] = sum;
				diff[1] = i;
				diff[2] = j;
			}
		}
	}
	*/
//==========================have a try===============================//
	int shang = sub_height >> 3;
	int mod = sub_height % 8;
	int subtemp_2;
	int graytemp_2;
	for (i = 0; i < gray_width - sub_width; i++)
	{
		for (j = 0; j < gray_height - sub_height; j++)
		{
			sum = 0;
			for (sub_i = 0; sub_i < sub_width; sub_i++)
			{
				subtemp_2 = sub_i*sub_height;
				graytemp_2 = (i + sub_i)*gray_height + j;
				for (sub_j = 0; sub_j < shang*8; sub_j = sub_j+8)
				{
					subtemp = subtemp_2+ sub_j;
					graytemp = graytemp_2 + sub_j;
					temp = subImg.data[subtemp] - grayImg.data[graytemp ];
					sum+=(temp >> 31)*(temp << 1) + temp;


					temp = subImg.data[subtemp+1] - grayImg.data[graytemp+1 ];
					sum += (temp >> 31)*(temp << 1) + temp;
				
					temp = subImg.data[subtemp +2] - grayImg.data[graytemp +2];
					sum += (temp >> 31)*(temp << 1) + temp;
					
					temp = subImg.data[subtemp +3] - grayImg.data[graytemp+3 ];
					sum += (temp >> 31)*(temp << 1) + temp;
					
					temp = subImg.data[subtemp +4] - grayImg.data[graytemp +4];
					sum += (temp >> 31)*(temp << 1) + temp;
					
					temp = subImg.data[subtemp +5] - grayImg.data[graytemp+5 ];
					sum += (temp >> 31)*(temp << 1) + temp;
					
					temp = subImg.data[subtemp +6] - grayImg.data[graytemp +6];
					sum += (temp >> 31)*(temp << 1) + temp;
					
					temp = subImg.data[subtemp +7] - grayImg.data[graytemp+7 ];
					sum += (temp >> 31)*(temp << 1) + temp;
				
				}
				
				for (sub_j= shang * 8; sub_j < sub_height; sub_j++)
				{
					temp = subImg.data[subtemp_2 + sub_j] - grayImg.data[graytemp_2 + sub_j];
					sum = sum + ((temp >> 31) & 1)*(0 - 2 * temp) + temp;
				}

			}
			if (sum < diff[0])
			{
				diff[0] = sum;
				diff[1] = i;
				diff[2] = j;
			}
		}
	}
	//============================================//

	if (diff[0] == sub_size * 255 )
	{
		cout << "not found" << endl;
		return(SUB_IMAGE_MATCH_FAIL);
	}
	
	*x = diff[2];
	*y = diff[1];
	return(SUB_IMAGE_MATCH_OK);

}

//************  task7  ************//
int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{

	if (colorImg.data == NULL||subImg.data==NULL)
	{
		cout << "error  input is null" << endl;
		return(SUB_IMAGE_MATCH_FAIL);
	}
	int sub_width = subImg.rows;
	int sub_height = subImg.cols;
	int color_width = colorImg.rows;
	int color_height = colorImg.cols;

	if (sub_width > color_width || sub_height > color_height)
	{
		cout << "error the size of subimag oversizes the source one  ";
		return(SUB_IMAGE_MATCH_FAIL);
	}

	int sub_size = sub_width*sub_height;
	int color_size = color_width*color_height;
    int sum_min=sub_size*255*3;
    int sum;
	int i, j,sub_i,sub_j;
	int diff[2];
	int temp;
	int subtemp_i, colortemp,subtemp_j;
	
	for (i = 0; i < color_width - sub_width; i++)
	{
		for (j = 0; j < color_height - sub_height; j++)
		{
			sum = 0;
			for (sub_i = 0; sub_i < sub_width; sub_i++)
			{
				subtemp_i = sub_i*sub_height*3;
				colortemp = ((i+sub_i)*color_height+j)*3;
				for (sub_j = 0; sub_j < sub_height; sub_j++)
				{
					subtemp_j = sub_j * 3;
					temp= ((subImg.data[subtemp_i+ subtemp_j] - colorImg.data[colortemp  + subtemp_j]) +
					      (subImg.data[subtemp_i + subtemp_j + 1] - colorImg.data[colortemp  + subtemp_j + 1]) +
					      (subImg.data[subtemp_i + subtemp_j + 2] - colorImg.data[colortemp  + subtemp_j + 2]));

					sum = sum + (temp >> 31)*(temp << 1) + temp;
				}
			}
			if (sum < sum_min)
			{
				sum_min = sum;
				diff[0] = i;
				diff[1] = j;	
			}
		}
	}
	if (sum_min == sub_size * 255 * 3)
	{
		cout << "not found" << endl;
		return(SUB_IMAGE_MATCH_FAIL);
	}
	*x = diff[1];
	*y = diff[0];
	return(SUB_IMAGE_MATCH_OK);
}

//************  task8  ************//
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{

	if (grayImg.data == NULL||subImg.data==NULL)
	{
		cout << "error  input is null" << endl;
		return(SUB_IMAGE_MATCH_FAIL);
	}
	int sub_width = subImg.rows;
	int sub_height = subImg.cols;
	int gray_width = grayImg.rows;
	int gray_height = grayImg.cols;

	if (sub_width > gray_width || sub_height > gray_height)
	{
		cout << "error the size of subimag oversizes the source one  ";
		return(SUB_IMAGE_MATCH_FAIL);
	}

	int sub_size = sub_width*sub_height;
	int gray_size = gray_width*gray_height;
	int i, j, sub_i, sub_j;
	long int sum_square_T=0;
	long int sum_square_S=0;
	long int sum_multi_ST=0;
	float correlation = 0;
	float max_correlation= 0.5;
	int diff[2] = {0,0};

	for (sub_i = 0; sub_i < sub_width; sub_i++)
	{
		for (sub_j = 0; sub_j < sub_height; sub_j++)
		{
			sum_square_T += subImg.data[sub_i*sub_height+sub_j]* subImg.data[sub_i*sub_height + sub_j];
		}
	}

	int graytemp, subtemp;

	for (i = 0; i < gray_width - sub_width; i++)
	{
		for (j = 0; j < gray_height - sub_height; j++)
		{
			correlation = 0;
			sum_square_S = 0;
			sum_multi_ST = 0;
			
			for (sub_i = 0; sub_i < sub_width; sub_i++)
			{
				subtemp = sub_i*sub_height;
				graytemp = (i + sub_i)*gray_height + j;

				for (sub_j = 0; sub_j < sub_height; sub_j++)
				{
					//sum_square_S += grayImg.data[(i + sub_i)*gray_height + j + sub_j] * grayImg.data[(i+sub_i)*gray_height + sub_j+j];
					//sum_multi_ST += grayImg.data[(i + sub_i)*gray_height + j + sub_j] * subImg.data[sub_i*sub_height + sub_j];
					sum_square_S += grayImg.data[graytemp+ sub_j] * grayImg.data[graytemp + sub_j];
					sum_multi_ST += grayImg.data[graytemp+ sub_j] * subImg.data[subtemp + sub_j];
				}
			}
			correlation = sum_multi_ST / (sqrt(sum_square_S)*sqrt(sum_square_T));
			if (correlation > max_correlation)
			{
			/*	if (correlation == 1)
				{
					*x = i; *y = j;
					return(1);
				 }*/
				max_correlation = correlation;
				diff[0] = i;
				diff[1] = j;
			}
		}
	}


	if (max_correlation == 0)
	{
		cout << "not found " << endl;
		return(SUB_IMAGE_MATCH_FAIL);
	}
/*	if (i == gray_width - sub_width&&j == gray_height - sub_height)
	{
		cout << "find but not the perfect match " << endl;
	}*/
	*x = diff[1];
	*y = diff[0];
	return(SUB_IMAGE_MATCH_OK);

   

}

//************  task9  ************//   
int   ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{

	if (grayImg.data == NULL||subImg.data==NULL)
	{
		cout << "error  input is null" << endl;
		return(SUB_IMAGE_MATCH_FAIL);
	}
	int sub_width = subImg.rows;
	int sub_height = subImg.cols;
	int gray_width = grayImg.rows;
	int gray_height = grayImg.cols;

	if (sub_width > gray_width || sub_height > gray_height)
	{
		cout << "error the size of subimag oversizes the source one  ";
		return(SUB_IMAGE_MATCH_FAIL);
	}

	Mat gradImg_x_gray, gradImg_y_gray;
	ustc_CalcGrad(grayImg, gradImg_x_gray, gradImg_y_gray);
	Mat gray_angleImg, gray_magImg;
	ustc_CalcAngleMag(gradImg_x_gray, gradImg_y_gray, gray_angleImg, gray_magImg);              //caculagte the angle of grayImg

	Mat gradImg_x_sub, gradImg_y_sub;
	ustc_CalcGrad(subImg, gradImg_x_sub, gradImg_y_sub);
	Mat sub_angleImg, sub_magImg;
	ustc_CalcAngleMag(gradImg_x_sub, gradImg_y_sub, sub_angleImg, sub_magImg);

	float* gray_ = (float*)gray_angleImg.data;
	float* sub_ = (float*)sub_angleImg.data;


	int diff[2] = { 0,0 };
	int sum,min_sum;
	min_sum = 2147483647;                 // 32 bits int max num 
	int i, j, sub_i, sub_j;
	int temp;
	int graytemp,subtemp;
	for (i = 0+1; i < gray_width - sub_width; i++)
	{
		for (j = 0+1; j < gray_height - sub_height; j++)
		{
			sum = 0;
			for (sub_i = 0+1; sub_i < sub_width-1; sub_i++)
			{
				graytemp = (i + sub_i)*gray_height + j;
				subtemp = sub_i*sub_height;
				for (sub_j = 0+1; sub_j < sub_height-1; sub_j++)
				{
					temp = (int)(gray_[graytemp + sub_j] - sub_[subtemp + sub_j]);
					sum = sum + (temp >> 31)*(temp << 1) + temp;
				}
			}
			if (sum < min_sum)
			{
				min_sum = sum;
				diff[0] = i;
				diff[1] = j;
			}
		}
	}
	if (min_sum == 2147483647)
	{
		cout << "not found" << endl;
		return(SUB_IMAGE_MATCH_FAIL);
	}
	*x = diff[1];
	*y = diff[0];
	return(SUB_IMAGE_MATCH_OK);

}

//************  task10  ************//   
int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{

	if (grayImg.data == NULL||subImg.data==NULL)
	{
		cout << "error  input is null" << endl;
		return(SUB_IMAGE_MATCH_FAIL);
	}
	int sub_width = subImg.rows;
	int sub_height = subImg.cols;
	int gray_width = grayImg.rows;
	int gray_height = grayImg.cols;

	Mat gradImg_x_gray, gradImg_y_gray;
	ustc_CalcGrad(grayImg, gradImg_x_gray, gradImg_y_gray);
	Mat gray_angleImg, gray_magImg;
	ustc_CalcAngleMag(gradImg_x_gray, gradImg_y_gray, gray_angleImg, gray_magImg);              //caculagte the angle of grayImg

	Mat gradImg_x_sub, gradImg_y_sub;
	ustc_CalcGrad(subImg, gradImg_x_sub, gradImg_y_sub);
	Mat sub_angleImg, sub_magImg;
	ustc_CalcAngleMag(gradImg_x_sub, gradImg_y_sub, sub_angleImg, sub_magImg);

	float* gray_ = (float*)gray_magImg.data;
	float* sub_ = (float*)sub_magImg.data;


	int diff[2] = { 0,0 };
	int sum, min_sum;
	min_sum = 2147483647;                 // 32 bits int max num 
	//cout << "min_sum" << (int)min_sum << endl;
	int i, j, sub_i, sub_j;
    int temp;
	int graytemp, subtemp;
	for (i = 0; i < gray_width - sub_width; i++)
	{
		for (j = 0; j < gray_height - sub_height; j++)
		{
			sum = 0;
			for (sub_i = 0+1; sub_i < sub_width-1; sub_i++)
			{
				graytemp = (i + sub_i)*gray_height + j;
				subtemp = sub_i*sub_height;
				for (sub_j = 0+1; sub_j < sub_height-1; sub_j++)
				{
					temp = (int)(gray_[graytemp + sub_j] - sub_[ subtemp + sub_j]);
					sum += ((temp >> 31) & 1)*(0 - 2 * temp) + temp;
					
				}
			}
			if (sum < min_sum)
			{
				min_sum = sum;
				diff[0] = i;
				diff[1] = j;
			/*	if (sum == 0)
				{
					*x = i;
					*y = j;
					return(1);
				}*/
			}
		}
	}

	if (min_sum == abs(0-1))
	{
		cout << "not found" << endl;
		return(SUB_IMAGE_MATCH_FAIL);
	}
/*	if (i == gray_width - sub_width&&j == gray_height - sub_height)
	{
		cout << "find but not the perfect match" << endl;
	}
	cout << "min" << min_sum << endl;*/
	*x = diff[1];
	*y = diff[0];
	return(SUB_IMAGE_MATCH_OK);


}

//************  task11  ************// 
int  ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{

	if (grayImg.data == NULL||subImg.data==NULL)
	{
		cout << "error  input is null" << endl;
		return(SUB_IMAGE_MATCH_FAIL);
	}
	int sub_width = subImg.rows;
	int sub_height = subImg.cols;
	int gray_width = grayImg.rows;
	int gray_height = grayImg.cols;
	int sub_size = sub_width*sub_height;
	int gray_size = gray_width*gray_height;
	int sub_hist[256] = { 0 };
	int gray_hist[256] = { 0 };
	int i, j, count, temp_i, temp_j;
	int des_axis[2] = { 1,1 };
 	ustc_CalcHist(subImg, sub_hist, 256);
	int sum_hist=0;
	int min_sum_hist = 255 * sub_size;


	int calcmulti;
	for (i = 0; i < gray_width - sub_width; i++)
	{
		for (j = 0; j < gray_height - sub_height; j++)
		{

			int gray_hist[256] = { 0 };                                     
		

			for (temp_i = 0; temp_i < sub_width; temp_i++)
			{
				calcmulti = (i + temp_i)*gray_height + j;
				for (temp_j = 0; temp_j < sub_height; temp_j++)
				{
					gray_hist[grayImg.data[calcmulti + temp_j]] += 1;
				}
			}
           
			sum_hist = 0;
			int temptemp;
			for (count= 0; count < 256; count++)
			{
				temptemp = gray_hist[count] - sub_hist[count];
				sum_hist +=  (temptemp >> 31)*(temptemp << 1) + temptemp;
			}

			if (sum_hist < min_sum_hist)
			{
			/*	if (sum_hist == 0)
				{
				
					*x = i;
					*y = j;
					return(1);
				}*/
				min_sum_hist = sum_hist;
				des_axis[0] = i;
				des_axis[1] = j;
			}
		}
	}

	if (min_sum_hist == 255 * sub_size)
	{
		std::cout << "not found" << endl;
		return(SUB_IMAGE_MATCH_FAIL);
	}

	/*if (i == gray_width - sub_width&&j == gray_height - sub_height)
	{
		cout << "find but not the  perfect natch" << endl;
	}*/
	*x = des_axis[1];
	*y = des_axis[0];
	return(SUB_IMAGE_MATCH_OK);
}





