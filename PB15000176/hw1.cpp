#include<SubImageMatch.h>

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (NULL == bgrImg.data || NULL == grayImg.data)
		return SUB_IMAGE_MATCH_FAIL;
	int rows = bgrImg.rows;
	int cols = bgrImg.cols;
	int blue[256],green[256],red[256];
	for (int i = 0;i < 256;i++)
	{
		blue[i] = i * 0.114f;
		green[i] = i * 0.587f;
		red[i] = i * 0.299f;
	}
	for (int i = 0;i < rows;i++)
	{
		uchar* p_bgr = bgrImg.ptr<uchar>(i);
		uchar* p_gray = grayImg.ptr<uchar>(i);
		for (int j = 0;j < cols;j++)
		{
			int b = *p_bgr;
			int g = *(p_bgr+1);
			int r = *(p_bgr+2);
			*p_gray = blue[b] + green[g] + red[r];
			p_bgr += 3;
			p_gray++;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	if (NULL == grayImg.data || NULL == gradImg_x.data || NULL == gradImg_y.data)
		return SUB_IMAGE_MATCH_FAIL;
	int rows = grayImg.rows;
	int cols = grayImg.cols;
	
	uchar* p1;
	uchar* p2 = grayImg.ptr<uchar>(0);
	uchar* p3 = grayImg.ptr<uchar>(1);

	for (int i = 1;i < rows-1;i++)
	{
		p1 = p2;
		p2 = p3;
		p3 = grayImg.ptr<uchar>(i + 1);
		float* p_x = gradImg_x.ptr<float>(i) + 1;
		float* p_y = gradImg_y.ptr<float>(i) + 1;
		for (int j = 1;j < cols-1;j++)
		{
			int grad_x1, grad_x2, grad_y1, grad_y2;
			grad_x1 = *(p1 - 1) + *(p3 - 1);
			grad_x2 = *(p1 + 1) + *(p3 + 1);
			grad_x1 += *(p2 - 1) << 1;
			grad_x2 += *(p2 + 1) << 1;
			p2++;
			grad_y1 = *(p1 - 1) + *(p1 + 1);
			grad_y2 = *(p3 - 1) + *(p3 + 1);
			grad_y1 += *p1 << 1;
			grad_y2 += *p3 << 1;
			p1++;
			p3++;
			*p_x = grad_x2 - grad_x1;
			*p_y = grad_y2 - grad_y1;
			p_x++;
			p_y++;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data || NULL == angleImg.data || NULL == magImg.data)
		return SUB_IMAGE_MATCH_FAIL;
	int rows = gradImg_x.rows;
	int cols = gradImg_x.cols;
	int ATan[1001];
	for (int i = 0;i < 1001;i++)
		ATan[i] = atan(i / 1000.0f) * 573;
	for (int i = 0;i < rows;i++)
	{
		for (int j = 0;j < cols;j++)
		{
			float grad_x = abs(((float*)gradImg_x.data)[i * cols + j]);
			float grad_y = abs(((float*)gradImg_y.data)[i * cols + j]);
			float angle;

			if (grad_x == 0 && grad_y == 0)
				angle = 0;
			else if (grad_x < grad_y)
				angle = 90 - ATan[(int)(grad_x / grad_y * 1000)] / 10.0f;
			else
				angle = ATan[(int)(grad_y / grad_x * 1000)] / 10.0f;

			if (grad_x >= 0 && grad_y <= 0)
				angle = 360 - angle;
			else if (grad_x <= 0 && grad_y >= 0)
				angle = 180 - angle;
			else
				angle += 180;
			float mag = sqrt(grad_x*grad_x + grad_y*grad_y);
			((float*)angleImg.data)[i * cols + j] = angle;
			((float*)magImg.data)[i * cols + j] = mag;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	if (NULL == grayImg.data || NULL == binaryImg.data)
		return SUB_IMAGE_MATCH_FAIL;

	int rows = grayImg.rows;
	int cols = grayImg.cols;
	for (int i = 0;i < rows;i++)
	{
		uchar* p_gray = grayImg.ptr<uchar>(i);
		uchar* p_binary = binaryImg.ptr<uchar>(i);
		for (int j = 0;j < cols;j++)
		{
			int gray = th - *p_gray;
			*p_binary = gray >> 31;
			p_gray++;
			p_binary++;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (NULL == grayImg.data || NULL == hist || hist_len < 256)
		return SUB_IMAGE_MATCH_FAIL;

	int cols = grayImg.cols;
	int rows = grayImg.rows;

	for (int i = 0; i < hist_len; i++)
	{
		hist[i] = 0;
	}

	for (int i = 0; i < rows; i++)
	{
		uchar* p_gray = grayImg.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			int pixVal = *p_gray;
			hist[pixVal]++;
			p_gray++;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data || NULL == x || NULL == y)
		return SUB_IMAGE_MATCH_FAIL;

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width < sub_width || height < sub_height)
		return SUB_IMAGE_MATCH_FAIL;

	int diff_min = INT32_MAX;
	int min_x, min_y;

	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			uchar* p_gray = grayImg.ptr<uchar>(i) + j;
			uchar* p_sub = subImg.ptr<uchar>(0);
			for (int m = 0; m < sub_height; m++)
			{
				for (int n = 0; n < sub_width; n++)
				{
					int bigImg_pix = *p_gray;
					int template_pix = *p_sub;
					total_diff += abs(bigImg_pix - template_pix);
					p_gray++;
					p_sub++;
				}
				p_gray += width - sub_width;
			}
			if (total_diff < diff_min)
			{
				diff_min = total_diff;
				min_x = i;
				min_y = j;
				if (diff_min == 0)
				{
					*x = min_x;
					*y = min_y;
					return SUB_IMAGE_MATCH_OK;
				}
			}
		}
	}
	*x = min_x;
	*y = min_y;
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (NULL == colorImg.data || NULL == subImg.data || NULL == x || NULL == y)
		return SUB_IMAGE_MATCH_FAIL;

	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width < sub_width || height < sub_height)
		return SUB_IMAGE_MATCH_FAIL;
	
	int diff_min = INT32_MAX;
	int min_x, min_y;

	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			uchar* p_color = colorImg.ptr<uchar>(i) + j * 3;
			uchar* p_sub = subImg.ptr<uchar>(0);
			for (int m = 0; m < sub_height; m++)
			{
				for (int n = 0; n < sub_width; n++)
				{
					int bigImg_pix = *p_color + *(p_color + 1) + *(p_color + 2);
					int template_pix = *p_sub + *(p_sub + 1) + *(p_sub + 2);
					total_diff += abs(bigImg_pix - template_pix);
					p_color+=3;
					p_sub+=3;
				}
				p_color += (width - sub_width) * 3;
			}
			if (total_diff < diff_min)
			{
				diff_min = total_diff;
				min_x = i;
				min_y = j;
				if (diff_min == 0)
				{
					*x = min_x;
					*y = min_y;
					return SUB_IMAGE_MATCH_OK;
				}
			}
		}
	}
	*x = min_x;
	*y = min_y;
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data || NULL == x || NULL == y)
		return SUB_IMAGE_MATCH_FAIL;

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width < sub_width || height < sub_height)
		return SUB_IMAGE_MATCH_FAIL;

	int diff_max = 0;
	int max_x, max_y;

	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			float total_diff = 0;
			int A = 0, B = 0, C = 0;
			uchar* p_gray = grayImg.ptr<uchar>(i) + j;
			uchar* p_sub = subImg.ptr<uchar>(0);
			for (int m = 0; m < sub_height; m++)
			{
				for (int n = 0; n < sub_width; n++)
				{
					int bigImg_pix = *p_gray;
					int template_pix = *p_sub;
					A += bigImg_pix * template_pix;
					B += bigImg_pix * bigImg_pix;
					C += template_pix * template_pix;
					p_gray++;
					p_sub++;
				}
				p_gray += width - sub_width;
			}
			total_diff = A / (sqrt(B)*sqrt(C));
			if (total_diff > diff_max)
			{
				diff_max = total_diff;
				max_x = i;
				max_y = j;
				if (diff_max == 1)
				{
					*x = max_x;
					*y = max_y;
					return SUB_IMAGE_MATCH_OK;
				}
			}
		}
	}
	*x = max_x;
	*y = max_y;
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data || NULL == x || NULL == y)
		return SUB_IMAGE_MATCH_FAIL;

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width < sub_width || height < sub_height)
		return SUB_IMAGE_MATCH_FAIL;

	int diff_min = INT32_MAX;
	int min_x, min_y;

	Mat angleImg(height, width, CV_8UC1);
	
	uchar* p1;
	uchar* p2 = grayImg.ptr<uchar>(0);
	uchar* p3 = grayImg.ptr<uchar>(1);
	for (int i = 1;i < height - 1;i++)
	{
		p1 = p2;
		p2 = p3;
		p3 = grayImg.ptr<uchar>(i + 1);
		uchar* p_angle = angleImg.ptr<uchar>(i) + 1;
		for (int j = 1;j < width - 1;j++)
		{
			int grad_x1, grad_x2, grad_y1, grad_y2;
			grad_x1 = *(p1 - 1) + *(p3 - 1);
			grad_x2 = *(p1 + 1) + *(p3 + 1);
			grad_x1 += *(p2 - 1) << 1;
			grad_x2 += *(p2 + 1) << 1;
			p2++;
			grad_y1 = *(p1 - 1) + *(p1 + 1);
			grad_y2 = *(p3 - 1) + *(p3 + 1);
			grad_y1 += *p1 << 1;
			grad_y2 += *p3 << 1;
			p1++;
			p3++;
			int p_x = grad_x2 - grad_x1;
			int p_y = grad_y2 - grad_y1;
			*p_angle = (uchar)(atan2(p_y, p_x) * 57.0f);
			p_angle++;
		}
	}

	Mat sub_angleImg(sub_height, sub_width, CV_8UC1);

	p2 = subImg.ptr<uchar>(0);
	p3 = subImg.ptr<uchar>(1);
	for (int i = 1;i < sub_height - 1;i++)
	{
		p1 = p2;
		p2 = p3;
		p3 = subImg.ptr<uchar>(i + 1);
		uchar* p_angle = sub_angleImg.ptr<uchar>(i) + 1;
		for (int j = 1;j < sub_width - 1;j++)
		{
			int grad_x1, grad_x2, grad_y1, grad_y2;
			grad_x1 = *(p1 - 1) + *(p3 - 1);
			grad_x2 = *(p1 + 1) + *(p3 + 1);
			grad_x1 += *(p2 - 1) << 1;
			grad_x2 += *(p2 + 1) << 1;
			p2++;
			grad_y1 = *(p1 - 1) + *(p1 + 1);
			grad_y2 = *(p3 - 1) + *(p3 + 1);
			grad_y1 += *p1 << 1;
			grad_y2 += *p3 << 1;
			p1++;
			p3++;
			int p_x = grad_x2 - grad_x1;
			int p_y = grad_y2 - grad_y1;
			*p_angle = (uchar)(atan2(p_y, p_x) * 57.0f);
			p_angle++;
		}
	}
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			uchar* p_gray = angleImg.ptr<uchar>(i) + j;
			uchar* p_sub = sub_angleImg.ptr<uchar>(0);
			for (int m = 0; m < sub_height; m++)
			{
				for (int n = 0; n < sub_width; n++)
				{
					int bigImg_pix = *p_gray;
					int template_pix = *p_sub;
					total_diff += abs(bigImg_pix - template_pix);
					p_gray++;
					p_sub++;
				}
				p_gray += width - sub_width;
			}
			if (total_diff < diff_min)
			{
				diff_min = total_diff;
				min_x = i;
				min_y = j;
				if (diff_min == 0)
				{
					*x = min_x;
					*y = min_y;
					return SUB_IMAGE_MATCH_OK;
				}
			}
		}
	}
	*x = min_x;
	*y = min_y;
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data || NULL == x || NULL == y)
		return SUB_IMAGE_MATCH_FAIL;

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width < sub_width || height < sub_height)
		return SUB_IMAGE_MATCH_FAIL;

	int diff_min = INT32_MAX;
	int min_x, min_y;

	Mat magImg(height, width, CV_8UC1);

	uchar* p1;
	uchar* p2 = grayImg.ptr<uchar>(0);
	uchar* p3 = grayImg.ptr<uchar>(1);
	for (int i = 1;i < height - 1;i++)
	{
		p1 = p2;
		p2 = p3;
		p3 = grayImg.ptr<uchar>(i + 1);
		uchar* p_angle = magImg.ptr<uchar>(i) + 1;
		for (int j = 1;j < width - 1;j++)
		{
			int grad_x1, grad_x2, grad_y1, grad_y2;
			grad_x1 = *(p1 - 1) + *(p3 - 1);
			grad_x2 = *(p1 + 1) + *(p3 + 1);
			grad_x1 += *(p2 - 1) << 1;
			grad_x2 += *(p2 + 1) << 1;
			p2++;
			grad_y1 = *(p1 - 1) + *(p1 + 1);
			grad_y2 = *(p3 - 1) + *(p3 + 1);
			grad_y1 += *p1 << 1;
			grad_y2 += *p3 << 1;
			p1++;
			p3++;
			int p_x = grad_x2 - grad_x1;
			int p_y = grad_y2 - grad_y1;
			*p_angle = (uchar)sqrt(p_x * p_x + p_y * p_y);
			p_angle++;
		}
	}

	Mat sub_magImg(sub_height, sub_width, CV_8UC1);

	p2 = subImg.ptr<uchar>(0);
	p3 = subImg.ptr<uchar>(1);
	for (int i = 1;i < sub_height - 1;i++)
	{
		p1 = p2;
		p2 = p3;
		p3 = subImg.ptr<uchar>(i + 1);
		uchar* p_angle = sub_magImg.ptr<uchar>(i) + 1;
		for (int j = 1;j < sub_width - 1;j++)
		{
			int grad_x1, grad_x2, grad_y1, grad_y2;
			grad_x1 = *(p1 - 1) + *(p3 - 1);
			grad_x2 = *(p1 + 1) + *(p3 + 1);
			grad_x1 += *(p2 - 1) << 1;
			grad_x2 += *(p2 + 1) << 1;
			p2++;
			grad_y1 = *(p1 - 1) + *(p1 + 1);
			grad_y2 = *(p3 - 1) + *(p3 + 1);
			grad_y1 += *p1 << 1;
			grad_y2 += *p3 << 1;
			p1++;
			p3++;
			int p_x = grad_x2 - grad_x1;
			int p_y = grad_y2 - grad_y1;
			*p_angle = (uchar)sqrt(p_x * p_x + p_y * p_y);
			p_angle++;
		}
	}
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			uchar* p_gray = magImg.ptr<uchar>(i) + j;
			uchar* p_sub = sub_magImg.ptr<uchar>(0);
			for (int m = 0; m < sub_height; m++)
			{
				for (int n = 0; n < sub_width; n++)
				{
					int bigImg_pix = *p_gray;
					int template_pix = *p_sub;
					total_diff += abs(bigImg_pix - template_pix);
					p_gray++;
					p_sub++;
				}
				p_gray += width - sub_width;
			}
			if (total_diff < diff_min)
			{
				diff_min = total_diff;
				min_x = i;
				min_y = j;
				if (diff_min == 0)
				{
					*x = min_x;
					*y = min_y;
					return SUB_IMAGE_MATCH_OK;
				}
			}
		}
	}
	*x = min_x;
	*y = min_y;
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data || NULL == x || NULL == y)
		return SUB_IMAGE_MATCH_FAIL;

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	if (width < sub_width || height < sub_height)
		return SUB_IMAGE_MATCH_FAIL;

	int diff_min = INT32_MAX;
	int min_x, min_y;
	int sub_hist[256] = { 0 };
	uchar* p = subImg.ptr<uchar>(0);
	for (int i = 0;i < sub_height*sub_width;i++)
	{
		sub_hist[*p]++;
		p++;
	}
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			int temp_hist[256] = { 0 };
			uchar* p_gray = grayImg.ptr<uchar>(i) + j;
			for (int m = 0; m < sub_height; m++)
			{
				for (int n = 0; n < sub_width; n++)
				{
					int gray = *p_gray;
					temp_hist[gray]++;
					p_gray++;
				}
				p_gray += width - sub_width;
			}
			for (int m = 0;m < 256;m++)
			{
				total_diff += abs(temp_hist[m] - sub_hist[m]);
			}
			if (total_diff < diff_min)
			{
				diff_min = total_diff;
				min_x = i;
				min_y = j;
				if (diff_min == 0)
				{
					*x = min_x;
					*y = min_y;
					return SUB_IMAGE_MATCH_OK;
				}
			}
		}
	}
	*x = min_x;
	*y = min_y;
	return SUB_IMAGE_MATCH_OK;
}
