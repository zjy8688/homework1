#include "SubImageMatch.h"
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (bgrImg.channels()!=3)
	{
		cout << "image is gray not bgr." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if( grayImg.channels() != 1)
	{
		cout << "image is bgr not gray." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = bgrImg.cols;
	int height = bgrImg.rows;
	if (width != grayImg.cols || height != grayImg.rows)
	{
		cout << "size not match." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int b, g, r,rw=-width,rwj=0;
	int grayVal;
	

	for (int row_i = 0; row_i < height; row_i++)
	{
		rw +=width;
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			rwj =3*(rw + col_j);
			b = bgrImg.data[rwj + 0];
			g = bgrImg.data[rwj + 1];
			r = bgrImg.data[rwj + 2];

			grayVal = b * 0.114f + g * 0.587f + r * 0.229f;
			grayImg.data[row_i * width + col_j] = grayVal;
		}
	}
	return SUB_IMAGE_MATCH_OK;

}

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th=50)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.channels() != 1)
	{
		cout << "image is bgr not gray." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (binaryImg.channels() != 1)
	{
		cout << "image is bgr not gray." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	if (width != binaryImg.cols || height != binaryImg.rows)
	{
		cout << "size not match." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int temp0, temp1, pixVal, dstVal;
	for (int row_i = 0; row_i < height; row_i++)
	{
		temp0 = row_i * width;
		for (int col_j = 0; col_j < width; col_j += 2)
		{
			//int pixVal = grayImg.at<uchar>(row_i, col_j);
			temp1 = temp0 + col_j;
			pixVal = grayImg.data[temp1];
			if (pixVal <th)
			{
				dstVal = 0;
			}
			else
			{
				dstVal =255;
			}
			//binaryImg.at<uchar>(row_i, col_j) = dstVal;
			binaryImg.data[temp1] = dstVal;
		}
	}


	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.channels() != 1)
	{
		cout << "image is bgr not gray." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	if (width != gradImg_x.cols || height != gradImg_x.rows)
	{
		cout << "size not match." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (width != gradImg_y.cols || height != gradImg_y.rows)
	{
		cout << "size not match." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int grad_x;
	int grad_y;
	int rw = 0,rwj1=0,rwj2=0;

	//计算x方向梯度图
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		rw = row_i * width;
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			rwj1 = rw + width + col_j+1;
			rwj2 = rw - width + col_j+1;
			    grad_x =
				grayImg.data[rwj2]
				+  grayImg.data[rw + col_j + 1]+ grayImg.data[rw + col_j + 1]
				+ grayImg.data[rwj1]
				- grayImg.data[rwj2-2]
				-  grayImg.data[rw + col_j - 1]- grayImg.data[rw + col_j - 1]
				- grayImg.data[rwj1-2];
				grad_y =
					grayImg.data[rwj1-2]
					+ 2 * grayImg.data[rwj1-1]
					+ grayImg.data[rwj1]
					- grayImg.data[rwj2-2]
					- 2 * grayImg.data[rwj2-1]
					- grayImg.data[rwj2];

				((float*)gradImg_y.data)[rw + col_j] = grad_y;

			((float*)gradImg_x.data)[rw + col_j] = grad_x;
		}
	}
	return SUB_IMAGE_MATCH_OK;



}


int  ustc_CalcAngleMag(Mat  gradImg_x, Mat gradImg_y, Mat&angleImg, Mat& magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (gradImg_y.channels() != 1)
	{
		cout << "image is bgr not gray." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (gradImg_x.channels() != 1)
	{
		cout << "image is bgr not gray." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (gradImg_y.cols != gradImg_x.cols || gradImg_y.rows != gradImg_x.rows)
	{
		cout << "size not match." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (gradImg_y.cols != angleImg.cols || angleImg.rows != gradImg_y.rows)
	{
		cout << "size not match." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (gradImg_y.cols != magImg.cols || magImg.rows != gradImg_y.rows)
	{
		cout << "size not match." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = gradImg_x.cols;
	int height = gradImg_x.rows;
	int rw=width,rwj = 0;
	float angle = 0.1;
	float mag = 0;
	float xy;
	float zhi[10000];
	for (int i = 0; i < 10000; i++)
	{
		zhi[i] = atan(float(-20.0 + (float)i*20 / 5000))/3.141592653*180;
	}

	//计算角度图
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			rw += 1;
			float grad_x = ((float*)gradImg_x.data)[rw];
			float grad_y = ((float*)gradImg_y.data)[rw];
			//mag = sqrt(grad_x*grad_x + grad_y*grad_y);
			mag = grad_x*grad_x + grad_y*grad_y;
			float xhalf = 0.5f*mag;
			int i = *(int*)&mag; // get bits for floating VALUE   
			i = 0x5f375a86 - (i >> 1); // gives initial guess y0  
			mag = *(float*)&i; // convert bits BACK to float  
			mag = mag*(1.5f - xhalf*mag*mag); // Newton step, repeating increases accuracy  
			mag = mag*(1.5f - xhalf*mag*mag); // Newton step, repeating increases accuracy  
			mag = mag*(1.5f - xhalf*mag*mag); // Newton step, repeating increases accuracy
			mag = 1/mag;
			if (grad_x > 0 && grad_y > 0)
			{
				xy = grad_y / grad_x;
				if (xy < 20)
					angle = zhi[int((xy + 20) / 0.004)];
				else
					angle = 90;
			}
			else if (grad_x < 0 && grad_y > 0)
			{
				xy = grad_y / grad_x;
				if (xy > -20)
					angle = zhi[int((xy + 20) / 0.004)] + 180;
				else
					angle = 90;
			}
			else if (grad_x < 0 && grad_y < 0)
			{
				xy = grad_y / grad_x;
				if (xy < 20)
					angle = zhi[int((xy + 20) / 0.004) ]-180;
				else
					angle = -90;
			}
			else if (grad_x > 0 && grad_y < 0)
			{
				xy = grad_y / grad_x;
				if (xy > -20)
					angle = zhi[int((xy + 20) / 0.004)];
				else
					angle = -90;
			}
			else if (grad_x == 0)
			{
				if (grad_y > 0)
					angle = 90;
				else if (grad_y < 0)
					angle = -90;
				else
					angle = 0;
			}
			else if(grad_x > 0)
			{
				angle = 0;
			}
			else
			{
				angle = 180;
			}


			
			//angle = angle / 180 * 3.141592653;
			//angle = atan2(grad_y, grad_x);
			//自己找办法优化三角函数速度，并且转化为角度制，规范化到0-360
			((float*)magImg.data)[rw] = mag;
			((float*)angleImg.data)[rw] = angle;
		}
	}
	return SUB_IMAGE_MATCH_OK;


}

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (NULL == grayImg.data || NULL == hist)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;

	//直方图清零
	for (int i = 0; i < hist_len; i++)
	{
		hist[i] = 0;
	}

	//计算直方图
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int pixVal = grayImg.data[row_i * width + col_j];
			hist[pixVal]++;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}


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
		cout << "too big for father img" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	float minxy=0;
	int x1, y1, x2, y2;
	int i, j;
	int row_index, col_index, bigImg_pix, template_pix;
	int total_diff = 0;

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	minxy = sub_width*sub_height * 25500000;
	int ij;
	int xw,xsw=0;
	int bt;
	for ( i = 0; i < height - sub_height; i++)
	{
		for ( j = 0; j < width - sub_width; j++)
		{
			total_diff = 0;
			//遍历模板图上的每一个像素
			ij = i*width + j;
			xw = ij;
			xsw = 0;
			for ( x1 = 0; x1 < sub_height; x1++)
			{
				xw += width;
				xsw += sub_width;
				for ( y1 = 0; y1 < sub_width; y1++)
				{
					//大图上的像素位置
					//row_index = i + x1;
					//col_index = j + y1;
					bigImg_pix = grayImg.data[xw+y1];
					template_pix = subImg.data[xsw + y1];
					bt = bigImg_pix - template_pix;
					int antisign_r = !((bt>> 31) & 1);
					total_diff +=(bt)*(2 * antisign_r - 1);
				}
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[i * width + j] = total_diff;
			if (total_diff < minxy)
			{
				minxy = total_diff;
				*x = i;
				*y = j;
			}
		}
	}
	/*Mat test(sub_height, sub_width, CV_8UC1);
	for (x1 = 0; x1 < sub_height; x1++)
	{
		for (y1 = 0; y1 < sub_width; y1++)
		{
			//大图上的像素位置
			int row_index = *x + x1;
			int col_index = *y + y1;
			test.data[x1 * sub_width + y1] = grayImg.data[row_index * width + col_index];

		}
	}
	imshow("imagetu", test);
	waitKey(0);*/
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y) {
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (colorImg.channels() != 3)
	{
		cout << "image is gray not bgr." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (subImg.channels() != 3)
	{
		cout << "image is gray not bgr." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (sub_width > width || sub_height > height)
	{
		cout << "too big for father img" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg = Mat::zeros(height - sub_height, width - sub_width, CV_32FC3);
	searchImg.setTo(FLT_MAX);
	//匹配误差初始化
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int row = 0; row < height - sub_height; row++)

	{
		for (int col = 0; col < width - sub_width; col++)
		{
			int total_diff_b = 0;
			int total_diff_g = 0;
			int total_diff_r = 0;
			//遍历模板图上的每一个像素
			for (int sub_row = 0; sub_row < sub_height; sub_row++)
			{
				for (int sub_col = 0; sub_col < sub_width; sub_col++)
				{
					//大图上的像素位置
					int row_index = row + sub_row;
					int col_index = col + sub_col;
					int placeindex = 3 * (row_index * width + col_index);
					int sub_place = 3 * (sub_row*sub_width + sub_col);
					int bigImg_pix_b = colorImg.data[placeindex];
					//模板图上的像素
					int template_pix_b = subImg.data[sub_place];
					int tmpdiffer_b = bigImg_pix_b - template_pix_b;
					int antisign_b = !((tmpdiffer_b >> 31) & 1);
					total_diff_b += (bigImg_pix_b - template_pix_b)*(2 * antisign_b - 1);
					int bigImg_pix_g = colorImg.data[placeindex + 1];
					//模板图上的像素
					int template_pix_g = subImg.data[sub_place + 1];
					int tmpdiffer_g = bigImg_pix_g - template_pix_g;
					int antisign_g = !((tmpdiffer_g >> 31) & 1);
					total_diff_g += (bigImg_pix_g - template_pix_g)*(2 * antisign_g - 1);
					int bigImg_pix_r = colorImg.data[placeindex + 2];
					//模板图上的像素
					int template_pix_r = subImg.data[sub_place + 2];
					int tmpdiffer_r = bigImg_pix_r - template_pix_r;
					int antisign_r = !((tmpdiffer_r >> 31) & 1);
					total_diff_r += (bigImg_pix_r - template_pix_r)*(2 * antisign_r - 1);

				}
				//存储当前像素位置的匹配误差
				((float*)searchImg.data)[3 * (row * (width - sub_width) + col)] = total_diff_b;
				((float*)searchImg.data)[3 * (row * (width - sub_width) + col) + 1] = total_diff_g;
				((float*)searchImg.data)[3 * (row * (width - sub_width) + col) + 2] = total_diff_r;
			}

		}
	}

	float tmp = 1000; *x = 0; *y = 0;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++) {
			float bgrplus = ((float*)searchImg.data)[3 * (i*(width - sub_width) + j)]
				+ ((float*)searchImg.data)[3 * (i*(width - sub_width) + j) + 1]
				+ ((float*)searchImg.data)[3 * (i*(width - sub_width) + j) + 2];
			if (bgrplus< tmp) {
				tmp = bgrplus;
				*x = j;
				*y = i;
			}

		}

	}
	return SUB_IMAGE_MATCH_OK;


}

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//int start = clock();
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	float mag = 0;
	int rc=0;
	int rcs;
	if (sub_width > width || sub_height > height)
	{
		cout << "too big for father img" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.channels() != 1)
	{
		cout << "image is gray not bgr." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (subImg.channels() != 1)
	{
		cout << "image is gray not bgr." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//该图用于记录每一个像素位置的R(i,j)
	Mat searchImg = Mat::zeros(height - sub_height, width - sub_width, CV_32FC1);
	searchImg.setTo(FLT_MAX);
	//匹配误差初始化
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int row = 0; row < height - sub_height; row++)

	{
		for (int col = 0; col < width - sub_width; col++)
		{
			float R = 0;
			float ST = 0;
			float SS = 0;
			float TT = 0;
			//遍历模板图上的每一个像素
			rc = row*width + col-width;
			rcs = -1;
			for (int sub_row = 0; sub_row < sub_height; sub_row++)
			{
				rc += width;
				for (int sub_col = 0; sub_col < sub_width; sub_col++)
				{
					rcs++;
					//大图上的像素位置
					//int row_index = row + sub_row;
					//int col_index = col + sub_col;
					int bigImg_pix = grayImg.data[rc+sub_col];
					//模板图上的像素
					int template_pix = subImg.data[rcs];
					ST += bigImg_pix*template_pix;
					TT += bigImg_pix*bigImg_pix;
					SS += template_pix*template_pix;
				}
				mag = TT*SS;
				float xhalf = 0.5f*mag;
				int i = *(int*)&mag; // get bits for floating VALUE   
				i = 0x5f375a86 - (i >> 1); // gives initial guess y0  
				mag = *(float*)&i; // convert bits BACK to float  
				mag = mag*(1.5f - xhalf*mag*mag); // Newton step, repeating increases accuracy  
				mag = mag*(1.5f - xhalf*mag*mag); // Newton step, repeating increases accuracy  
				mag = mag*(1.5f - xhalf*mag*mag); // Newton step, repeating increases accuracy
				mag = 1 / mag;
				R = ST /mag;
				//存储当前像素位置的匹配误差
				((float*)searchImg.data)[row * (width - sub_width) + col] = R;
			}

		}
	}

	float tmp = 0.8; *x = 0; *y = 0;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++) {
			if (((float*)searchImg.data)[i*(width - sub_width) + j] > tmp) {
				tmp = ((float*)searchImg.data)[i*(width - sub_width) + j];
				*x = j;
				*y = i;
			}

		}

	}
	return SUB_IMAGE_MATCH_OK;
}
int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y) 
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.channels() != 1)
	{
		cout << "image is gray not bgr." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (subImg.channels() != 1)
	{
		cout << "image is gray not bgr." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (subImg.rows > grayImg.rows || subImg.cols > grayImg.cols)
	{
		cout << "too big for father img" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;
	Mat sub_gradImg_x(sub_height, sub_width, CV_32FC1);
	Mat sub_gradImg_y(sub_height, sub_width, CV_32FC1);
	Mat sub_angleImg(sub_height, sub_width, CV_32FC1);
	Mat sub_magImg(sub_height, sub_width, CV_32FC1);
	ustc_CalcGrad(subImg, sub_gradImg_x, sub_gradImg_y);
	ustc_CalcAngleMag(sub_gradImg_x, sub_gradImg_y, sub_angleImg, sub_magImg);

	int height = grayImg.rows;
	int width = grayImg.cols;
	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);
	Mat angleImg(height, width, CV_32FC1);
	Mat magImg(height, width, CV_32FC1);
	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);
	Mat searchImg(height - sub_height, width - sub_width, CV_32FC1);
	searchImg.setTo(FLT_MAX);
	int rc=0;
	for (int row = 1; row < height - sub_height - 1; row++)

	{
		for (int col = 1; col < width - sub_width - 1; col++)
		{
			rc = row*width + col;
			float total_diff = 0;
			//遍历模板图上的每一个像素
			for (int sub_row = 1; sub_row < sub_height - 1; sub_row++)
			{
				rc += width;
				for (int sub_col = 1; sub_col < sub_width - 1; sub_col++)
				{
					//大图上的像素位置
					//int row_index = row + sub_row;
					//int col_index = col + sub_col;
					float bigImg_pix = ((float*)angleImg.data)[rc + sub_col];
					//模板图上的像素
					float template_pix = ((float*)sub_angleImg.data)[sub_row * sub_width + sub_col];
					int tmpdiffer = (int)(bigImg_pix - template_pix);
					int antisign = !((tmpdiffer >> 31) & 1);
					//total_diff += abs(bigImg_pix - template_pix);
					total_diff += (bigImg_pix - template_pix)*(2 * antisign - 1);
				}
				//存储当前像素位置的匹配误差
				((float*)searchImg.data)[row * (width - sub_width) + col] = total_diff;
			}

		}
	}

	float tmp = 9999999999999999999; *x = 0; *y = 0;
	for (int i = 1; i < height - sub_height - 1; i++)
	{
		for (int j = 1; j < width - sub_width - 1; j++) {
			if (((float*)searchImg.data)[i*(width - sub_width) + j] < tmp) {
				tmp = ((float*)searchImg.data)[i*(width - sub_width) + j];
				*x = j;
				*y = i;
			}

		}

	}
	return SUB_IMAGE_MATCH_OK;



}
int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	//int start = clock();
	if (grayImg.channels() != 1)
	{
		cout << "image is gray not bgr." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (subImg.channels() != 1)
	{
		cout << "image is gray not bgr." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (subImg.rows > grayImg.rows || subImg.cols > grayImg.cols)
	{
		cout << "too big for father img" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int hist_len = 256;
	int sub_hist[256] = { 0 };
	int tmp_hist[256] = { 0 };
	ustc_CalcHist(subImg, sub_hist, 256);
	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg = Mat::zeros(height - sub_height, width - sub_width, CV_32FC1);
	searchImg.setTo(FLT_MAX);
	//匹配误差初始化
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int row = 0; row < height - sub_height; row++)

	{
		for (int col = 0; col < width - sub_width; col++)
		{
			int total_diff = 0;
			//直方图清零
			for (int i = 0; i < hist_len; i++)
			{
				tmp_hist[i] = 0;
			}
			//计算直方图
			for (int row_i = 0; row_i < sub_height; row_i++)
			{
				for (int col_j = 0; col_j < sub_width; col_j += 1)
				{
					int rowindex = row + row_i;
					int colindex = col + col_j;
					int pixVal = grayImg.data[rowindex * width + colindex];
					tmp_hist[pixVal]++;
				}
			}
			for (int num = 0; num < hist_len; num++) {
				int subpix = sub_hist[num];
				int tmppix = tmp_hist[num];
				int tmpdiffer = subpix - tmppix;
				int antisign = !((tmpdiffer >> 31) & 1);
				total_diff += (subpix - tmppix)*(2 * antisign - 1);
			}
			((float*)searchImg.data)[row * (width - sub_width) + col] = total_diff;
		}

	}

	float tmp = 9999999999999999999; *x = 0; *y = 0;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++) {
			if (((float*)searchImg.data)[i*(width - sub_width) + j] < tmp) {
				tmp = ((float*)searchImg.data)[i*(width - sub_width) + j];
				*x = j;
				*y = i;
			}

		}


	}
	return SUB_IMAGE_MATCH_OK;
}
int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int sub_height = subImg.rows;
	int sub_width = subImg.cols;
	Mat sub_gradImg_x(sub_height, sub_width, CV_32FC1);
	Mat sub_gradImg_y(sub_height, sub_width, CV_32FC1);
	Mat sub_angleImg(sub_height, sub_width, CV_32FC1);
	Mat sub_magImg(sub_height, sub_width, CV_32FC1);
	ustc_CalcGrad(subImg, sub_gradImg_x, sub_gradImg_y);
	ustc_CalcAngleMag(sub_gradImg_x, sub_gradImg_y, sub_angleImg, sub_magImg);
	int height = grayImg.rows;
	int width = grayImg.cols;
	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);
	Mat angleImg(height, width, CV_32FC1);
	Mat magImg(height, width, CV_32FC1);

	ustc_CalcGrad(grayImg, gradImg_x, gradImg_y);
	ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg);
	Mat searchImg(height - sub_height, width - sub_width, CV_32FC1);
	searchImg.setTo(FLT_MAX);
	for (int row = 1; row < height - sub_height - 1; row++)

	{
		for (int col = 1; col < width - sub_width - 1; col++)
		{
			float total_diff = 0;
			//遍历模板图上的每一个像素
			for (int sub_row = 1; sub_row < sub_height - 1; sub_row++)
			{
				for (int sub_col = 1; sub_col < sub_width - 1; sub_col++)
				{
					//大图上的像素位置
					int row_index = row + sub_row;
					int col_index = col + sub_col;
					float bigImg_pix = ((float*)magImg.data)[row_index * width + col_index];
					//模板图上的像素
					float template_pix = ((float*)sub_magImg.data)[sub_row * sub_width + sub_col];
					int tmpdiffer = (int)(bigImg_pix - template_pix);
					int antisign = !((tmpdiffer >> 31) & 1);
					total_diff += (bigImg_pix - template_pix)*(2 * antisign - 1);
				}
				//存储当前像素位置的匹配误差
				((float*)searchImg.data)[row * (width - sub_width) + col] = total_diff;
			}

		}
	}

	float tmp = 9999999999999999999; *x = 0; *y = 0;
	for (int i = 1; i < height - sub_height - 1; i++)
	{
		for (int j = 1; j < width - sub_width - 1; j++) {
			if (((float*)searchImg.data)[i*(width - sub_width) + j] < tmp) {
				tmp = ((float*)searchImg.data)[i*(width - sub_width) + j];
				*x = i;
				*y = j;
			}

		}

	}
	return SUB_IMAGE_MATCH_OK;
}
