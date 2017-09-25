#include "SubImageMatch.h"

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat &grayImg)
{
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = bgrImg.cols;
	int height = bgrImg.rows;

	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int b = bgrImg.data[3 * (row_i * width + col_j) + 0];
			int g = bgrImg.data[3 * (row_i * width + col_j) + 1];
			int r = bgrImg.data[3 * (row_i * width + col_j) + 2];
			int grayVal = b * 0.114f + g * 0.587f + r * 0.229f;
			grayImg.data[row_i * width + col_j] = grayVal;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcGrad(Mat grayImg, Mat &gradImg_x, Mat &gradImg_y)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;



	for (int row_i = 1; row_i < height - 1; row_i += 1)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int grad_x =
				grayImg.data[(row_i - 1) * width + col_j + 1]
				+ 2 * grayImg.data[(row_i)* width + col_j + 1]
				+ grayImg.data[(row_i + 1)* width + col_j + 1]
				- grayImg.data[(row_i - 1) * width + col_j - 1]
				- 2 * grayImg.data[(row_i)* width + col_j - 1]
				- grayImg.data[(row_i + 1)* width + col_j - 1];

			((float*)gradImg_x.data)[row_i * width + col_j] = grad_x;

		}
	}


	for (int row_i = 1; row_i < height - 1; row_i += 1)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int grad_y =
				-grayImg.data[(row_i - 1) * width + col_j - 1]
				- 2 * grayImg.data[(row_i - 1)* width + col_j]
				- grayImg.data[(row_i - 1)* width + col_j + 1]
				+ grayImg.data[(row_i + 1) * width + col_j - 1]
				+ 2 * grayImg.data[(row_i + 1)* width + col_j]
				+ grayImg.data[(row_i + 1)* width + col_j + 1];

			((float*)gradImg_y.data)[row_i * width + col_j] = grad_y;
		}
	}


#ifdef IMG_SHOW
	Mat gradImg_x_8U(height, width, CV_8UC1);
	
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int val = ((float*)gradImg_x.data)[row_i * width + col_j];
			gradImg_x_8U.data[row_i * width + col_j] = abs(val);
		}
	}

	Mat gradImg_y_8U(height, width, CV_8UC1);
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int val = ((float*)gradImg_y.data)[row_i * width + col_j];
			gradImg_y_8U.data[row_i * width + col_j] = abs(val);
		}
	}


	namedWindow("gradImg_x_8U", 0);
	imshow("gradImg_x_8U", gradImg_x_8U);
	namedWindow("gradImg_y_8U", 0);
	imshow("gradImg_y_8U", gradImg_x_8U);
	waitKey();
#endif
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg) {
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = gradImg_x.cols;
	int height = gradImg_x.rows;

	angleImg.setTo(0);
	magImg.setTo(0);


	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			float grad_x = ((float*)gradImg_x.data)[row_i * width + col_j];
			float grad_y = ((float*)gradImg_y.data)[row_i * width + col_j];
			float angle = atan2(grad_y, grad_x);
			float mag = (float)sqrt(grad_x*grad_x + grad_y*grad_y);
			if (angle < 0) { angle = angle + CV_2PI; }
			angle = angle / CV_PI * 180;
			((float*)angleImg.data)[row_i * width + col_j] = angle;
			((float*)magImg.data)[row_i*width + col_j] = mag;
		}
	}

#ifdef IMG_SHOW
	Mat angleImg_8U(height, width, CV_8UC1);

	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			float angle = ((float*)angleImg.data)[row_i * width + col_j];
	
			angle /= 2;
			angleImg_8U.data[row_i * width + col_j] = angle;
		}
	}
	namedWindow("angleImg_8U", 0);
	imshow("angleImg_8U", angleImg_8U);
	waitKey();
#endif
	return SUB_IMAGE_MATCH_OK;
}

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th) {
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;

	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{

			int temp1 = row_i * width + col_j;
			int pixVal = grayImg.data[temp1];
			int dstVal = 0;
			if (pixVal >= th)
			{
				dstVal = 255;
			}
			else if (pixVal < th)
			{
				dstVal = 0;
			}

			binaryImg.data[temp1] = dstVal;
		}
	}

#ifdef IMG_SHOW
	namedWindow("binaryImg", 0);
	imshow("binaryImg", binaryImg);
	waitKey();
#endif

	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len) {
	if (NULL == grayImg.data || NULL == hist)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;

	for (int i = 0; i < hist_len; i++)
	{
		hist[i] = 0;
	}

	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int pixVal = grayImg.data[row_i * width + col_j];
			if (pixVal < hist_len)
			{
				hist[pixVal]++;
			}
			else { printf("The point(%d,%d) is too bright too be included", row_i, col_j); }
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y) {

	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width < sub_width || height < sub_height)
	{
		cout << "two images' sizes are not suitable." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	Mat searchImg(height, width, CV_32FC1);
	searchImg.setTo(FLT_MAX);

	for (int i = 0; i < (height - sub_height); i++)
	{
		for (int j = 0; j < (width - sub_width); j++)
		{
			int total_diff = 0;
			for (int m = 0; m < sub_height; m++)
			{
				for (int n = 0; n < sub_width; n++)
				{
					int row_index = i + m;
					int col_index = j + n;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					int template_pix = subImg.data[m * sub_width + n];

					total_diff += abs(bigImg_pix - template_pix);
				}
			}
			((float*)searchImg.data)[i * width + j] = total_diff;
		}
	}


	float flag = FLT_MAX;
	for (int i = 0; i <= height - sub_height; i++)
	{
		for (int j = 0; j <= width - sub_width; j++)
		{
			if (((float*)searchImg.data)[i*width + j] < flag)
			{
				*x = j;
				*y = i;
				flag = ((float*)searchImg.data)[i*width + j];
			}
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

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

	if (width < sub_width || height < sub_height)
	{
		cout << "two images' sizes are not suitable." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	Mat searchImg(height, width, CV_32FC1);
	searchImg.setTo(FLT_MAX);

	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			for (int m = 0; m < sub_height; m++)
			{
				for (int n = 0; n < sub_width; n++)
				{
					int row_index = i + m;
					int col_index = j + n;
					int b = colorImg.data[3 * (row_index * width + col_index) + 0];
					int g = colorImg.data[3 * (row_index * width + col_index) + 1];
					int r = colorImg.data[3 * (row_index * width + col_index) + 2];
					int bigImg_pix = b + g + r;
					int sub_b = subImg.data[3 * (m*sub_width + n) + 0];
					int sub_g = subImg.data[3 * (m*sub_width + n) + 1];
					int sub_r = subImg.data[3 * (m*sub_width + n) + 2];
					int template_pix = sub_b + sub_g + sub_r;

					total_diff += abs(bigImg_pix - template_pix);
				}
			}
			((float*)searchImg.data)[i * width + j] = total_diff;
		}
	}

	float flag = FLT_MAX;
	for (int i = 0; i <= height - sub_height; i++)
	{
		for (int j = 0; j <= width - sub_width; j++)
		{
			if (((float*)searchImg.data)[i*width + j] < flag)
			{
				*x = j;
				*y = i;
				flag = ((float*)searchImg.data)[i*width + j];
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

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (width < sub_width-1 || height < sub_height-1)
	{
		cout << "two images' sizes are not suitable." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	Mat searchImg(height, width, CV_32FC1);
	searchImg.setTo(FLT_MAX);

	for (int i = 0; i < (height - sub_height); i++)
	{
		for (int j = 0; j < (width - sub_width); j++)
		{
			int relation = 0;
			int Top = 0;
			int D_a = 0, D_b = 0;
			for (int m = 0; m < sub_height; m++)
			{
				for (int n = 0; n < sub_width; n++)
				{
					int row_index = i + m;
					int col_index = j + n;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					int template_pix = subImg.data[m * sub_width + n];

					Top += (bigImg_pix*template_pix);
					D_a += (bigImg_pix*bigImg_pix);
					D_b += (template_pix*template_pix);
				}
			}
			relation = Top / (sqrt(D_a)*sqrt(D_b));
			((float*)searchImg.data)[i * width + j] = relation;
		}
	}

	float flag = -1;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			if (((float*)searchImg.data)[i*width + j] > flag)
			{
				*x = j;
				*y = i;
				flag = ((float*)searchImg.data)[i*width + j];
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

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (width < sub_width || height < sub_height)
	{
		cout << "two images' sizes are not suitable." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	Mat searchImg(height, width, CV_32FC1);
	searchImg.setTo(FLT_MAX);

	for (int i = 0; i < (height - sub_height - 1); i++)
	{
		for (int j = 0; j < (width - sub_width - 1); j++)
		{
			int total_diff = 0;
			for (int m = 1; m < sub_height - 1; m++)
			{
				for (int n = 1; n < sub_width - 1; n++)
				{
					int row_index = i + m;
					int col_index = j + n;
					int gradImg_big_x =
						grayImg.data[(row_index - 1) * width + col_index + 1]
						+ 2 * grayImg.data[(row_index)* width + col_index + 1]
						+ grayImg.data[(row_index + 1)* width + col_index + 1]
						- grayImg.data[(row_index - 1) * width + col_index - 1]
						- 2 * grayImg.data[(row_index)* width + col_index - 1]
						- grayImg.data[(row_index + 1)* width + col_index - 1];
					int gradImg_sub_x =
						subImg.data[(m - 1) * sub_width + n + 1]
						+ 2 * subImg.data[(m)* sub_width + n + 1]
						+ subImg.data[(m + 1)* sub_width + n + 1]
						- subImg.data[(m - 1) * sub_width + n - 1]
						- 2 * subImg.data[(m)* sub_width + n - 1]
						- subImg.data[(m + 1)* sub_width + n - 1];
					int gradImg_big_y =
						-grayImg.data[(row_index - 1) * width + col_index - 1]
						- 2 * grayImg.data[(row_index - 1) * width + col_index]
						- grayImg.data[(row_index - 1)* width + col_index + 1]
						+ grayImg.data[(row_index + 1) * width + col_index - 1]
						+ 2 * grayImg.data[(row_index + 1) * width + col_index]
						+ grayImg.data[(row_index + 1) * width + col_index + 1];
					int gradImg_sub_y =
						-subImg.data[(m - 1) * sub_width + n - 1]
						- 2 * subImg.data[(m - 1) * sub_width + n]
						- subImg.data[(m - 1)* sub_width + n + 1]
						+ subImg.data[(m + 1) * sub_width + n - 1]
						+ 2 * subImg.data[(m + 1) * sub_width + n]
						+ subImg.data[(m + 1) * sub_width + n + 1];
					float angle_big = atan2(gradImg_big_y, gradImg_big_x);
					if (angle_big < 0)
					{
						angle_big = angle_big + CV_2PI;
					}
					float angle_sub = atan2(gradImg_sub_y, gradImg_sub_x);
					if (angle_sub < 0)
					{
						angle_sub = angle_sub + CV_2PI;
					}
					float cha = abs(angle_big - angle_sub);
					cha = cha / CV_PI * 180;
					if (cha > 180) { cha = 360 - cha; }
					total_diff = total_diff + cha;
				}
			}
			((float*)searchImg.data)[i * width + j] = total_diff;
		}
	}


	float flag = FLT_MAX;
	for (int i = 0; i < height - sub_height - 1; i++)
	{
		for (int j = 0; j < width - sub_width - 1; j++)
		{
			if (((float*)searchImg.data)[i*width + j] < flag)
			{
				*x = j;
				*y = i;
				flag = ((float*)searchImg.data)[i*width + j];
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

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (width < sub_width || height < sub_height)
	{
		cout << "two images' sizes are not suitable." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	Mat searchImg(height, width, CV_32FC1);
	searchImg.setTo(FLT_MAX);

	for (int i = 0; i < (height - sub_height - 1); i++)
	{
		for (int j = 0; j < (width - sub_width - 1); j++)
		{
			int total_diff = 0;
			for (int m = 1; m < sub_height - 1; m++)
			{
				for (int n = 1; n < sub_width - 1; n++)
				{
					int row_index = i + m;
					int col_index = j + n;
					int gradImg_big_x =
						grayImg.data[(row_index - 1) * width + col_index + 1]
						+ 2 * grayImg.data[(row_index)* width + col_index + 1]
						+ grayImg.data[(row_index + 1)* width + col_index + 1]
						- grayImg.data[(row_index - 1) * width + col_index - 1]
						- 2 * grayImg.data[(row_index)* width + col_index - 1]
						- grayImg.data[(row_index + 1)* width + col_index - 1];
					int gradImg_sub_x =
						subImg.data[(m - 1) * sub_width + n + 1]
						+ 2 * subImg.data[(m)* sub_width + n + 1]
						+ subImg.data[(m + 1)* sub_width + n + 1]
						- subImg.data[(m - 1) * sub_width + n - 1]
						- 2 * subImg.data[(m)* sub_width + n - 1]
						- subImg.data[(m + 1)* sub_width + n - 1];
					int gradImg_big_y =
						-grayImg.data[(row_index - 1) * width + col_index - 1]
						- 2 * grayImg.data[(row_index - 1) * width + col_index]
						- grayImg.data[(row_index - 1)* width + col_index + 1]
						+ grayImg.data[(row_index + 1) * width + col_index - 1]
						+ 2 * grayImg.data[(row_index + 1) * width + col_index]
						+ grayImg.data[(row_index + 1) * width + col_index + 1];
					int gradImg_sub_y =
						-subImg.data[(m - 1) * sub_width + n - 1]
						- 2 * subImg.data[(m - 1) * sub_width + n]
						- subImg.data[(m - 1)* sub_width + n + 1]
						+ subImg.data[(m + 1) * sub_width + n - 1]
						+ 2 * subImg.data[(m + 1) * sub_width + n]
						+ subImg.data[(m + 1) * sub_width + n + 1];
					int mag_big = sqrt(gradImg_big_x*gradImg_big_x + gradImg_big_y*gradImg_big_y);
					int mag_sub = sqrt(gradImg_sub_x*gradImg_sub_x + gradImg_sub_y*gradImg_sub_y);
					total_diff = total_diff + abs(mag_big - mag_sub);
				}
			}
			((float*)searchImg.data)[i * width + j] = total_diff;
		}
	}


	float flag = FLT_MAX;
	for (int i = 0; i < height - sub_height - 1; i++)
	{
		for (int j = 0; j < width - sub_width - 1; j++)
		{
			if (((float*)searchImg.data)[i*width + j] < flag)
			{
				*x = j;
				*y = i;
				flag = ((float*)searchImg.data)[i*width + j];
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

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (width < sub_width || height < sub_height)
	{
		cout << "two images' sizes are not suitable." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int grayImg_hist[500];
	int subImg_hist[500];
	int hist_length = 500;
	for (int i = 0; i < hist_length; i++)
	{
		grayImg_hist[i] = 0; 
		subImg_hist[i] = 0;
	}

	Mat searchImg(height, width, CV_32FC1);
	searchImg.setTo(FLT_MAX);

	for (int i = 0; i < (height - sub_height ); i++)
	{
		for (int j = 0; j < (width - sub_width ); j++)
		{
			int total_diff = 0;
			for (int i = 0; i < hist_length; i++)
			{
				grayImg_hist[i] = 0;
				subImg_hist[i] = 0;
			}
			for (int m = 0; m < sub_height; m++)
			{
				for (int n = 0; n < sub_width; n++)
				{
					int row_index = i + m;
					int col_index = j + n;
					int gray_pixVal = grayImg.data[row_index*width + col_index];
					if (gray_pixVal < hist_length)
					{
						grayImg_hist[gray_pixVal]++;
					}
					else
					{
						printf("The point (%d,%d) is too bright too be included.", row_index, col_index);
					}
					int sub_pixVal = subImg.data[m*sub_width + n];
					if (sub_pixVal < hist_length)
					{
						subImg_hist[sub_pixVal]++;
					}
					else
					{
						printf("The point (%d,%d) is too bright too be included", m, n);
					}
				}
			}
			for (int k = 0; k < hist_length; k++)
			{
				total_diff = total_diff + abs(subImg_hist[k] - grayImg_hist[k]);
			}
			((float*)searchImg.data)[i * width + j] = total_diff;
		}
	}
	float flag = FLT_MAX;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			if (((float*)searchImg.data)[i*width + j] < flag)
			{
				*x = j;
				*y = i;
				flag = ((float*)searchImg.data)[i*width + j];
			}
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

