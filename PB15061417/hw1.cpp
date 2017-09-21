#include "SubImageMatch.h"

//1
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = bgrImg.cols;
	int height = bgrImg.rows;
	grayImg = Mat(height, width, CV_8UC1, Scalar(0));

	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp1 = row_i *width;
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int temp2 = temp1 + col_j;
			uchar b = bgrImg.data[3 * temp2 + 0];
			uchar g = bgrImg.data[3 * temp2 + 1];
			uchar r = bgrImg.data[3 * temp2 + 2];
            uchar grayVal = b * 114 / 1000 + g * 587 /1000 + r * 229 / 1000;
			grayImg.data[temp2] = grayVal;
         }
	}



	return SUB_IMAGE_MATCH_OK;
}



//2
int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;

	gradImg_x = Mat(height, width, CV_32FC1, Scalar(0));
	gradImg_y = Mat(height, width, CV_32FC1, Scalar(0));
	

	//计算x,y方向梯度图
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int a11 = row_i * width + col_j,
				a10 = a11 - 1,
				a12 = a11 + 1,
				a00 = a10 - width,
				a01 = a00 + 1,
				a02 = a01 + 1,
				a20 = a10 + width,
				a21 = a20 + 1,
				a22 = a21 + 1;

			int grad_x =
				grayImg.data[a02]
				+ 2 * grayImg.data[a12]
				+ grayImg.data[a22]
				- grayImg.data[a00]
				- 2 * grayImg.data[a10]
				- grayImg.data[a20];
			int grad_y = 
				grayImg.data[a20]
				+ 2 * grayImg.data[a21]
				+ grayImg.data[a22]
				- grayImg.data[a00]
				- 2 * grayImg.data[a01]
				- grayImg.data[a02];


			((float*)gradImg_x.data)[a11] = grad_x;
			((float*)gradImg_y.data)[a11] = grad_y;
            
		}
	}

	return SUB_IMAGE_MATCH_OK;
}




//3*
int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = gradImg_x.cols;
	int height = gradImg_x.rows;

	angleImg = Mat(height, width, CV_32FC1, Scalar(0));
	magImg = Mat(height, width, CV_32FC1, Scalar(0));
	

	//计算角度图
	for (int row_i = 1; row_i < height - 1; row_i++)
	{   
		int temp0 = row_i * width;
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int temp1 = temp0 + col_j;
			float grad_x = ((float*)gradImg_x.data)[temp1];
			float grad_y = ((float*)gradImg_y.data)[temp1];
			
			
			float temp_angle = atan2(grad_y, grad_x);
			float angle = temp_angle / 6.2832 * 360;
			if (angle < 0)
			{
				angle = angle + 360;
			}
			
			//float mag = sqrt(grad_x * grad_x + grad_y * grad_y);
			float temp_x = grad_x * grad_x + grad_y * grad_y;
			float xhalf = 0.5f*temp_x;
			int s = *(int*)&temp_x; // get bits for floating VALUE 
			s = 0x5f375a86 - (s >> 1); // gives initial guess y0
			temp_x = *(float*)&s; // convert bits BACK to float
			temp_x = temp_x*(1.5f - xhalf*temp_x*temp_x); // Newton step, repeating increases accuracy
			float mag = 1 / temp_x;
			//自己找办法优化三角函数速度，并且转化为角度制，规范化到0-360
			((float*)angleImg.data)[temp1] = angle;
			((float*)magImg.data)[temp1] = mag;
			
		}
	}

	return SUB_IMAGE_MATCH_OK;
}


	
//4
int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	binaryImg = Mat(height, width, CV_8UC1, Scalar(0));

	int dst[256];
	for (int i = 0; i < 255; i++)
	{
		dst[i] = 255;
		if (i - th >> 31) dst[i] = 0;
	}
	
	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp0 = row_i * width;
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int temp1 = temp0 + col_j;
			int pixVal = grayImg.data[temp1];
			binaryImg.data[temp1] = dst[pixVal];
			
		}
	}


	return SUB_IMAGE_MATCH_OK;
}



//5
int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (NULL == grayImg.data || NULL == hist)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	if(hist_len != 256)
	{
		cout << "hist is WRONG." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;

	//直方图清零
	for (int i = 0; i < 256; i+=8)
	{
		hist[i] = 0;
		hist[i+1] = 0;
		hist[i+2] = 0;
		hist[i+3] = 0;
		hist[i+4] = 0;
		hist[i+5] = 0;
		hist[i+6] = 0;
		hist[i+7] = 0;
	}

	//计算直方图
	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp0 = row_i * width;
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int temp1 = temp0 + col_j;
			int pixVal = grayImg.data[temp1];
			hist[pixVal]++;
			
		}
	}

	return SUB_IMAGE_MATCH_OK;
}

//6
int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (sub_width > width || sub_height > height)
	{
		cout << "subimage is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int temp_i = 0;
	int temp_j = 0;
	int temp_min = INT_MAX;

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int ax = 0; ax < sub_height; ax++)
			{
				for (int ay = 0; ay < sub_width; ay++)
				{
					//大图上的像素位置
					int row_index = i + ay;
					int col_index = j + ax;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					//模板图上的像素
					int template_pix = subImg.data[ay * sub_width + ax];
					int temp_abs = bigImg_pix - template_pix;
					//加绝对值
					total_diff += (((temp_abs >> 31) & 1)*(0 - 2 * temp_abs) + temp_abs);
					//total_diff += abs(bigImg_pix - template_pix);
				}
			}
			
			if (total_diff < temp_min)
			{
				temp_min = total_diff;
				temp_i = i;
				temp_j = j;
			}
		}
	}
	*x = temp_j;
	*y = temp_i;

	return SUB_IMAGE_MATCH_OK;
	
}



//7

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (sub_width > width || sub_height > height)
	{
		cout << "subimage is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int temp_i = 0;
	int temp_j = 0;
	int temp_min = INT_MAX;

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int ax = 0; ax < sub_height; ax++)
			{
				for (int ay = 0; ay < sub_width; ay++)
				{
					//大图上的像素位置
					int row_index = i + ay;
					int col_index = j + ax;
					int temp1 = (row_index *  width + col_index) * 3;
					int bigImg_pix_0 = colorImg.data[temp1 + 0];
					int bigImg_pix_1 = colorImg.data[temp1 + 1];
					int bigImg_pix_2 = colorImg.data[temp1 + 2];
					//模板图上的像素
					int temp2 = (ay * sub_width + ax) * 3;
					int template_pix_0 = subImg.data[temp2 + 0];
					int template_pix_1 = subImg.data[temp2 + 1];
					int template_pix_2 = subImg.data[temp2 + 2];

					//加绝对值
					int temp_abs = bigImg_pix_0 + bigImg_pix_1 + bigImg_pix_2 - template_pix_0 - template_pix_1 - template_pix_2;
					total_diff += (((temp_abs >> 31) & 1)*(0 - 2 * temp_abs) + temp_abs);
				}
			}
			
			if (total_diff < temp_min)
			{
				temp_min = total_diff;
				temp_i = i;
				temp_j = j;
			}
		}
	}
	*x = temp_j;
	*y = temp_i;

	return SUB_IMAGE_MATCH_OK;
}




//8
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (sub_width > width || sub_height > height)
	{
		cout << "subimage is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int temp_i = 0;
	int temp_j = 0;
	float temp_max = 0;
	float temp_r;

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			long long nume = 0;
			long long deno_1 = 0;
			long long deno_2 = 0;
			//遍历模板图上的每一个像素
			for (int ax = 0; ax < sub_height; ax++)
			{
				for (int ay = 0; ay < sub_width; ay++)
				{
					//大图上的像素位置
					int row_index = i + ay;
					int col_index = j + ax;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					//p1 = p + row_index * width + col_index;
					//模板图上的像素
					int template_pix = subImg.data[ay * sub_width + ax];
					//q1 = q + ay * sub_width + ax;

					nume += (bigImg_pix * template_pix);
					deno_1 += (bigImg_pix * bigImg_pix);
				    deno_2 += (template_pix * template_pix);
				}
			}
			
			float temp_x = deno_1 * deno_2;
			//float temp_y = sqrt(temp_x);
			//求平方根的倒数
			float xhalf = 0.5f*temp_x;
			int s = *(int*)&temp_x; // get bits for floating VALUE 
			s = 0x5f375a86 - (s >> 1); // gives initial guess y0
			temp_x = *(float*)&s; // convert bits BACK to float
			temp_x = temp_x*(1.5f - xhalf*temp_x*temp_x); // Newton step, repeating increases accuracy
			

			temp_r = nume * temp_x;
			if (temp_r > temp_max)
			{
				temp_max = temp_r;
				temp_i = i;
				temp_j = j;
			}
		}
	}

	*x = temp_j;
	*y = temp_i;
	//cout << temp_max << endl;
	return SUB_IMAGE_MATCH_OK;
}


//9
int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (sub_width > width || sub_height > height)
	{
		cout << "subimage is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}


    Mat grayGrad_x(height, width, CV_32FC1); Mat grayGrad_y(height, width, CV_32FC1);
	int flag1 = ustc_CalcGrad(grayImg, grayGrad_x, grayGrad_y);
	if(NULL == flag1)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	Mat grayAngle(height, width, CV_32FC1); Mat grayMag(height, width, CV_32FC1);
	int flag2 = ustc_CalcAngleMag(grayGrad_x, grayGrad_y, grayAngle, grayMag);
	if (NULL == flag2)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	
	Mat subGrad_x(sub_height, sub_width, CV_32FC1); Mat subGrad_y(sub_height, sub_width, CV_32FC1);
	int flag3 = ustc_CalcGrad(subImg, subGrad_x, subGrad_y);
	if (NULL == flag3)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	Mat subAngle(sub_height, sub_width, CV_32FC1); Mat subMag(sub_height, sub_width, CV_32FC1);
	int flag4 = ustc_CalcAngleMag(subGrad_x, subGrad_y, subAngle, subMag);
	if (NULL == flag4)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	
	
	if (NULL == grayAngle.data || NULL == subAngle.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int temp_i = 0;
	int temp_j = 0;
	int temp_min = INT_MAX;/////

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int ax = 0; ax < sub_height; ax++)
			{
				for (int ay = 0; ay < sub_width; ay++)
				{
					//大图上的像素位置
					int row_index = i + ay;
					int col_index = j + ax;
					
					float bigImg_pix = ((float*)grayAngle.data)[row_index * width + col_index];
					//模板图上的像素
					float template_pix = ((float*)subAngle.data)[ay * sub_width + ax];
					int temp_abs = (int)(bigImg_pix - template_pix);
					int temp_diff = ((temp_abs >> 31) & 1)*(0 - 2 * temp_abs) + temp_abs;
					//加绝对值
					if (temp_diff > 180)
					{
						temp_diff = 360 - temp_diff;
					}
					total_diff += temp_diff;
					//total_diff += abs(bigImg_pix - template_pix);
				}
			}
			//存储当前像素位置的匹配误差
			if (total_diff < temp_min)
			{
				temp_min = total_diff;
				temp_i = i;
				temp_j = j;
			}
		}
	}
	*x = temp_j;
	*y = temp_i;


	return SUB_IMAGE_MATCH_OK;
}

//10
int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (sub_width > width || sub_height > height)
	{
		cout << "subimage is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}


	Mat grayGrad_x(height, width, CV_32FC1); Mat grayGrad_y(height, width, CV_32FC1);
	int flag1 = ustc_CalcGrad(grayImg, grayGrad_x, grayGrad_y);
	if (NULL == flag1)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	Mat grayAngle(height, width, CV_32FC1); Mat grayMag(height, width, CV_32FC1);
	int flag2 = ustc_CalcAngleMag(grayGrad_x, grayGrad_y, grayAngle, grayMag);
	if (NULL == flag2)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	
	Mat subGrad_x(sub_height, sub_width, CV_32FC1); Mat subGrad_y(sub_height, sub_width, CV_32FC1);
	int flag3 = ustc_CalcGrad(subImg, subGrad_x, subGrad_y);
	if (NULL == flag3)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	Mat subAngle(sub_height, sub_width, CV_32FC1); Mat subMag(sub_height, sub_width, CV_32FC1);
	int flag4 = ustc_CalcAngleMag(subGrad_x, subGrad_y, subAngle, subMag);
	if (NULL == flag4)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}


	if (NULL == grayAngle.data || NULL == subAngle.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int temp_i = 0;
	int temp_j = 0;
	int temp_min = INT_MAX;/////

						   //遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int ax = 0; ax < sub_height; ax++)
			{
				for (int ay = 0; ay < sub_width; ay++)
				{
					//大图上的像素位置
					int row_index = i + ay;
					int col_index = j + ax;

					float bigImg_pix = ((float*)grayMag.data)[row_index * width + col_index];
					//模板图上的像素
					float template_pix = ((float*)subMag.data)[ay * sub_width + ax];
					int temp_abs = (int)(bigImg_pix - template_pix);
					int temp_diff = ((temp_abs >> 31) & 1)*(0 - 2 * temp_abs) + temp_abs;
					//加绝对值
					
					total_diff += temp_diff;
					
				}
			}
			//存储当前像素位置的匹配误差
			if (total_diff < temp_min)
			{
				temp_min = total_diff;
				temp_i = i;
				temp_j = j;
			}
		}
	}
	*x = temp_j;
	*y = temp_i;


	return SUB_IMAGE_MATCH_OK;
}

//11
int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (sub_width > width || sub_height > height)
	{
		cout << "subimage is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int temp_i = 0;
	int temp_j = 0;
	int temp_min = INT_MAX;

	int hist_len = 256;
	int sub_hist[256];
	ustc_CalcHist(subImg, sub_hist, hist_len);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	int* hist = new int[hist_len];
	memset(hist, 0, sizeof(int) * hist_len);

	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			//清零
			memset(hist, 0, sizeof(int) * hist_len);

			//计算当前位置直方图
			for (int x = 0; x < sub_height; x++)
			{
				for (int y = 0; y < sub_width; y++)
				{
					//大图上的像素位置
					int row_index = i + y;
					int col_index = j + x;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					hist[bigImg_pix]++;
				}
			}

			//根据直方图计算匹配误差
			int total_diff = 0;
			for (int ii = 0; ii < hist_len; ii++)
			{
				//total_diff += abs(hist[ii] - sub_hist[ii]);
				int temp_abs = hist[ii] - sub_hist[ii];
				total_diff += (((temp_abs >> 31) & 1)*(0 - 2 * temp_abs) + temp_abs);
			}
			//存储当前像素位置的匹配误差
			//(float*)searchImg.data)[i * width + j] = total_diff;
			if (total_diff < temp_min)
			{
				temp_min = total_diff;
				temp_i = i;
				temp_j = j;
			}

		}
	}

	delete[] hist;
	*x = temp_j;
	*y = temp_i;
	return SUB_IMAGE_MATCH_OK;
}


