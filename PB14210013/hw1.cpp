#include "SubImageMatch.h"
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return  SUB_IMAGE_MATCH_FAIL;
		;
	}

	int width = bgrImg.cols;
	int height = bgrImg.rows;


	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp0 = row_i * width;
		for (int col_j = 0; col_j < width; col_j++)
		{
			int temp1 = row_i * width + col_j;
			int temp2 = 3 * temp1;
			int b = bgrImg.data[temp2 + 0];
			int g = bgrImg.data[temp2 + 1];
			int r = bgrImg.data[temp2 + 2];
			int grayVal = (b * 114 + g * 587 + r * 229) / 1000;
			grayImg.data[temp1] = (grayVal);
		}
	}
#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey();
#endif

	return SUB_IMAGE_MATCH_OK;

}

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y){
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return  SUB_IMAGE_MATCH_FAIL;
		;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int grad_x, grad_y;

	//����x�����ݶ�ͼ
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		int temp0 = row_i*width;
		for (int col_j = 1; col_j < width - 1;)
		{
			grad_x =
				grayImg.data[temp0 - width + col_j + 1]
				+ 2 * grayImg.data[temp0 + col_j + 1]
				+ grayImg.data[temp0 + width + col_j + 1]
				- grayImg.data[temp0 - width + col_j - 1]
				- 2 * grayImg.data[temp0 + col_j - 1]
				- grayImg.data[temp0 + width + col_j - 1];


			((float*)gradImg_x.data)[temp0 + col_j] = grad_x;


			col_j += 1;

		}
	}

	//����y�����ݶ�ͼ
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		int temp0 = row_i*width;
		for (int col_j = 1; col_j < width - 1;)
		{
			grad_y =
				-grayImg.data[temp0 - width + col_j - 1]
				- 2 * grayImg.data[temp0 - width + col_j]
				- grayImg.data[temp0 - width + col_j + 1]
				+ grayImg.data[temp0 + width + col_j - 1]
				+ 2 * grayImg.data[temp0 + width + col_j]
				+ grayImg.data[temp0 + width + col_j + 1];

			((float*)gradImg_y.data)[temp0 + col_j] = grad_y;
			col_j += 1;

		}
	}


#ifdef IMG_SHOW
	Mat gradImg_x_8U(height, width, CV_8UC1);
	//Ϊ�˷����۲죬ֱ��ȡ����ֵ
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int val = ((float*)gradImg_x.data)[row_i * width + col_j];
			gradImg_x_8U.data[row_i * width + col_j] = abs(val);
		}
	}

	namedWindow("gradImg_x_8U", 0);
	imshow("gradImg_x_8U", gradImg_x_8U);
	waitKey();

	Mat gradImg_y_8U(height, width, CV_8UC1);
	//Ϊ�˷����۲죬ֱ��ȡ����ֵ
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int val = ((float*)gradImg_y.data)[row_i * width + col_j];
			gradImg_y_8U.data[row_i * width + col_j] = abs(val);
		}
	}

	namedWindow("gradImg_y_8U", 0);
	imshow("gradImg_y_8U", gradImg_y_8U);
	waitKey();
#endif
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = gradImg_x.cols;
	int height = gradImg_x.rows;



	//�����Ƕ�ͼ
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			float grad_x = ((float*)gradImg_x.data)[row_i * width + col_j];
			float grad_y = ((float*)gradImg_y.data)[row_i * width + col_j];
			//float angle = atan2(grad_y, grad_x);

			float ax = abs(grad_x), ay = abs(grad_y);
			float a = min(ax, ay) / (max(ax, ay) + (float)DBL_EPSILON);
			float a2 = a*a;
			float r = ((-0.0464964749 * a2 + 0.15931422) * a2 - 0.327622764) * a2 * a + a;
			if (ay > ax) r = 1.57079637 - r;
			if (grad_x < 0) r = 3.14159274f - r;
			if (grad_y < 0) r = 6.28318548f - r;


			((float*)angleImg.data)[row_i * width + col_j] = r / 180 * 3.1415926525f;
			float f1 = grad_x*grad_x + grad_y*grad_y;
			int t = *(int*)&f1;
			t -= 0x3f800000;
			t >>= 1;
			t += 0x3f800000;

			((float*)magImg.data)[row_i * width + col_j] = *(float*)&t;
		}
	}

#ifdef IMG_SHOW
	Mat angleImg_8U(height, width, CV_8UC1);
	//Ϊ�˷����۲죬����Щ���仯
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			float angle = ((float*)angleImg.data)[row_i * width + col_j];
			angle *= 180 / CV_PI;
			angle += 180;
			//Ϊ������8U����ʾ����С��0-180֮��
			angle /= 2;
			angleImg_8U.data[row_i * width + col_j] = angle;
		}
	}

	namedWindow("gradImg_x_8U", 0);
	imshow("gradImg_x_8U", angleImg_8U);
	waitKey();
#endif
}

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	int width = grayImg.cols;
	int height = grayImg.rows;

	int table[255];
	for (int i = 0; i < th; i++)
	{
		table[i] = 0;
	}
	for (int i = th; i <255; i++)
	{
		table[i] = 255;
	}
	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp0 = row_i * width;
		for (int col_j = 0; col_j < width; col_j += 1)
		{

			int temp1 = temp0 + col_j;
			int pixVal = grayImg.data[temp1];
			int dstVal;
			binaryImg.data[temp1] = table[pixVal];

		}
	}

#ifdef IMG_SHOW
	namedWindow("binaryImg", 0);
	imshow("binaryImg", binaryImg);
	waitKey();
#endif

	return SUB_IMAGE_MATCH_OK;

}

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{

	if (NULL == grayImg.data || NULL == hist)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;


	for (int i = 0; i < hist_len; i++)
	{
		hist[i] = 0;
	}


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


int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{

	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;



	int min = 0;
	for (int i = 0; i < sub_height; i++)
	{
		for (int j = 0; j< sub_width; j++)
		{

			int row_index = 0 + i;
			int col_index = 0 + j;
			int bigImg_pix = grayImg.data[row_index * width + col_index];
			//ģ��ͼ�ϵ�����
			int template_pix = subImg.data[i* sub_width + j];
			int aa = bigImg_pix - template_pix;
			int bb = aa >> 31;
			int cc = (aa ^ bb) - bb;
			min += cc;



		}
	}

	//������ͼÿһ�����أ�ע�����е���ʼ����ֹ����
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			//����ģ��ͼ�ϵ�ÿһ������
			for (int x0 = 0; x0 < sub_height; x0++)
			{
				for (int y0 = 0; y0 < sub_width; y0++)
				{

					//��ͼ�ϵ�����λ��
					int row_index = i + y0;
					int col_index = j + x0;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					//ģ��ͼ�ϵ�����
					int template_pix = subImg.data[y0 * sub_width + x0];



					total_diff += abs(bigImg_pix - template_pix);
				}
			}


			if (total_diff<min)
			{
				min = total_diff;
				*x = j;
				*y = i;
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
	int min = 0;
	int bigImg_b, bigImg_g, bigImg_r, sub_r, sub_g, sub_b, tmp_r, tmp_g, tmp_b;
	for (int i = 0; i < sub_height; i++)
	{

		for (int j = 0; j < sub_width; j++)
		{
			bigImg_b = colorImg.data[(i*width + j) * 3];
			bigImg_g = colorImg.data[(i*width + j) * 3 + 1];
			bigImg_r = colorImg.data[(i*width + j) * 3 + 2];
			//ģ��ͼ�ϵ�����
			sub_b = subImg.data[(i*sub_width + j) * 3];
			sub_g = subImg.data[(i*sub_width + j) * 3 + 1];
			sub_r = subImg.data[(i*sub_width + j) * 3 + 2];


			min += abs(bigImg_b - sub_b) + abs(bigImg_g - sub_g) + abs(bigImg_r - sub_r);

		}
	}
	int total_diff;

	*x = 0;
	*y = 0;

	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			total_diff = 0;
			for (int x0 = 0; x0 < sub_height; x0++)
			{

				for (int y0 = 0; y0 < sub_width; y0++)
				{
					bigImg_b = colorImg.data[((i + x0)*width + j + y0) * 3];
					bigImg_g = colorImg.data[((i + x0)*width + j + y0) * 3 + 1];
					bigImg_r = colorImg.data[((i + x0)*width + j + y0) * 3 + 2];
					sub_b = subImg.data[(x0*width + y0) * 3];
					sub_g = subImg.data[(x0*width + y0) * 3 + 1];
					sub_r = subImg.data[(x0*width + y0) * 3 + 2];



					total_diff += abs(bigImg_b - sub_b) + abs(bigImg_g - sub_g) + abs(bigImg_r - sub_r);

				}
			}
			if (total_diff<min)
			{
				min = total_diff;
				*x = j;
				*y = i;
			}
		}
	}
	return SUB_IMAGE_MATCH_OK;

}


int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int temp1, temp2, temp3;
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int  k2, bigImg_pix, template_pix;
	temp1 = 0;
	temp2 = 0;
	temp3 = 0;
	for (int i = 0; i < sub_height; i++)
	{

		for (int j = 0; j< sub_width; j++)
		{
			bigImg_pix = grayImg.data[i*width + j];
			//ģ��ͼ�ϵ�����
			template_pix = subImg.data[i*sub_width + j];
			temp1 += bigImg_pix*template_pix;
			temp2 += bigImg_pix*bigImg_pix;
			temp3 += template_pix*template_pix;
		}
	}
	float min = 1.0f*temp1 / (sqrt(temp2)*sqrt(temp3));
	*x = 0;
	*y = 0;


	for (int i = 0; i< height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			float diff = 0;
			temp1 = 0;
			temp2 = 0;
			temp3 = 0;
			for (int x0 = 0; x0 < sub_height; x0++)
			{

				for (int y0 = 0; y0 < sub_width; y0++)
				{
					bigImg_pix = grayImg.data[(i + x0)*width + j + y0];
					template_pix = subImg.data[x0*width + y0];
					temp1 += bigImg_pix*template_pix;
					temp2 += bigImg_pix*bigImg_pix;
					temp3 += template_pix*template_pix;
				}
			}
			diff = 1.0f*temp1 / (sqrt(temp2)*sqrt(temp3));
			if (diff>min)
			{
				min = diff;
				*x = j;
				*y = i;
			}
		}
	}
	if (*x >= 0 && *y >= 0)
	{
		return SUB_IMAGE_MATCH_OK;
	}
	else
	{
		return SUB_IMAGE_MATCH_FAIL;
	}

}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{


	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int gray_width = grayImg.cols;
	int gray_height = grayImg.rows;
	Mat gray_gradImg_x(gray_height, gray_width, CV_32FC1);
	Mat gray_gradImg_y(gray_height, gray_width, CV_32FC1);
	gray_gradImg_x.setTo(0);
	gray_gradImg_y.setTo(0);
	int flag1 = ustc_CalcGrad(grayImg, gray_gradImg_x, gray_gradImg_y);

	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	Mat sub_gradImg_x(sub_height, sub_width, CV_32FC1);
	Mat sub_gradImg_y(sub_height, sub_width, CV_32FC1);
	sub_gradImg_x.setTo(0);
	sub_gradImg_y.setTo(0);
	int flag2 = ustc_CalcGrad(subImg, sub_gradImg_x, sub_gradImg_y);


	Mat gray_angleImg(gray_height, gray_width, CV_32FC1);
	gray_angleImg.setTo(0);
	Mat gray_magImg(gray_height, gray_width, CV_32FC1);
	gray_magImg.setTo(0);
	int flag3 = ustc_CalcAngleMag(gray_gradImg_x, gray_gradImg_y, gray_angleImg, gray_magImg);

	Mat sub_angleImg(sub_height, sub_width, CV_32FC1);
	sub_angleImg.setTo(0);
	Mat sub_magImg(sub_height, sub_width, CV_32FC1);
	sub_magImg.setTo(0);
	int flag4 = ustc_CalcAngleMag(sub_gradImg_x, sub_gradImg_y, sub_angleImg, sub_magImg);



	float min = 0;
	for (int i = 0; i < sub_height; i++)
	{
		for (int j = 0; j< sub_width; j++)
		{

			int row_index = 0 + i;
			int col_index = 0 + j;



			float bigImg_pix = ((float*)gray_angleImg.data)[i* gray_width + j];
			float template_pix = ((float*)sub_angleImg.data)[i* sub_width + j];


			//ģ��ͼ�ϵ�����


			min += abs(bigImg_pix - template_pix);


		}
	}

	for (int i = 0; i < gray_height - sub_height; i++)
	{
		for (int j = 0; j < gray_width - sub_width; j++)
		{
			float total_diff = 0;
			//����ģ��ͼ�ϵ�ÿһ������
			for (int x0 = 0; x0 < sub_height; x0++)
			{
				for (int y0 = 0; y0 < sub_width; y0++)
				{

					//��ͼ�ϵ�����λ��
					int row_index = i + y0;
					int col_index = j + x0;




					float f1 = ((float*)gray_angleImg.data)[row_index * gray_width + col_index];
					float f2 = ((float*)sub_angleImg.data)[y0 * sub_width + x0];

					total_diff += abs(f1 - f2);
				}
			}


			if (total_diff<min)
			{
				min = total_diff;
				*x = j;
				*y = i;
			}

		}
	}
	if (*x >= 0 && *y >= 0)
	{
		return SUB_IMAGE_MATCH_OK;
	}
	else
	{
		return SUB_IMAGE_MATCH_FAIL;
	}


}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{

	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	int gray_width = grayImg.cols;
	int gray_height = grayImg.rows;
	Mat gray_gradImg_x(gray_height, gray_width, CV_32FC1);
	Mat gray_gradImg_y(gray_height, gray_width, CV_32FC1);
	gray_gradImg_x.setTo(0);
	gray_gradImg_y.setTo(0);
	int flag1 = ustc_CalcGrad(grayImg, gray_gradImg_x, gray_gradImg_y);

	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	Mat sub_gradImg_x(sub_height, sub_width, CV_32FC1);
	Mat sub_gradImg_y(sub_height, sub_width, CV_32FC1);
	sub_gradImg_x.setTo(0);
	sub_gradImg_y.setTo(0);
	int flag2 = ustc_CalcGrad(subImg, sub_gradImg_x, sub_gradImg_y);


	Mat gray_angleImg(gray_height, gray_width, CV_32FC1);
	gray_angleImg.setTo(0);
	Mat gray_magImg(gray_height, gray_width, CV_32FC1);
	gray_magImg.setTo(0);
	int flag3 = ustc_CalcAngleMag(gray_gradImg_x, gray_gradImg_y, gray_angleImg, gray_magImg);

	Mat sub_angleImg(sub_height, sub_width, CV_32FC1);
	sub_angleImg.setTo(0);
	Mat sub_magImg(sub_height, sub_width, CV_32FC1);
	sub_magImg.setTo(0);
	int flag4 = ustc_CalcAngleMag(sub_gradImg_x, sub_gradImg_y, sub_angleImg, sub_magImg);



	float min = 0;
	for (int i = 0; i < sub_height; i++)
	{

		for (int j = 0; j< sub_width; j++)
		{

			int row_index = 0 + i;
			int col_index = 0 + j;

			float bigImg_pix = ((float*)gray_magImg.data)[i* gray_width + j];
			float template_pix = ((float*)sub_magImg.data)[i* sub_width + j];


			//ģ��ͼ�ϵ�����


			min += abs(bigImg_pix - template_pix);


		}
	}

	for (int i = 0; i < gray_height - sub_height; i++)
	{
		for (int j = 0; j < gray_width - sub_width; j++)
		{
			float total_diff = 0;
			//����ģ��ͼ�ϵ�ÿһ������
			for (int x0 = 0; x0 < sub_height; x0++)
			{
				for (int y0 = 0; y0 < sub_width; y0++)
				{

					//��ͼ�ϵ�����λ��
					int row_index = i + y0;
					int col_index = j + x0;

					float f1 = ((float*)gray_magImg.data)[row_index * gray_width + col_index];
					float f2 = ((float*)sub_magImg.data)[y0 * sub_width + x0];

					total_diff += abs(f1 - f2);
				}
			}


			if (total_diff<min)
			{
				min = total_diff;
				*x = j;
				*y = i;
			}

		}
	}

	if (*x >= 0 && *y >= 0)
	{
		return SUB_IMAGE_MATCH_OK;
	}
	else
	{
		return SUB_IMAGE_MATCH_FAIL;
	}
}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{

	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return MY_FAIL;
	}



	int grayHist[256], subHist[256];
	int flag1 = ustc_CalcHist(subImg, subHist, 256);
	long d0 = 0x7fffffff;
	int i0, j0;
	for (int i = 0; i <= grayImg.rows - subImg.rows; i++)
	{
		for (int j = 0; j < grayImg.cols - subImg.cols; j++)
		{
			int flag2 = ustc_CalcHist(grayImg(Rect(j, i, subImg.cols, subImg.rows)).clone(), grayHist, 256);
			long d = 0;

			for (int k = 0; k < 256; k++)
			{
				d += abs(grayHist[k] - subHist[k]);
			}
			if (d < d0)
			{
				d0 = d;
				*x = j;
				*y = i;
			}
		}
	}

	if (*x >= 0 && *y >= 0)
	{
		return SUB_IMAGE_MATCH_OK;
	}
	else
	{
		return SUB_IMAGE_MATCH_FAIL;
	}

}
