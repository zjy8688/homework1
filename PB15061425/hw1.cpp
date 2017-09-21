#include "SubImageMatch.h"

//1.彩色转灰度
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (nullptr == bgrImg.data || nullptr == grayImg.data)
	{
		std::cout << "1.image is null!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (bgrImg.rows != grayImg.rows || bgrImg.cols != grayImg.cols)
	{
		std::cout << "1.sizes don't match!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}

	int i = bgrImg.cols*bgrImg.rows - 1, j = 0;
	int b, g, r;
	for (;i >= 0;i--)
	{
		j = 3 * i;
		*(grayImg.data + i) = (*(bgrImg.data + j) * 116 + *(bgrImg.data + j + 1) * 601 + *(bgrImg.data + j + 2) * 306) >> 10;
	}
	
	return SUB_IMAGE_MATCH_OK;
}

//2.计算梯度
int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	if (nullptr == grayImg.data || nullptr == gradImg_x.data || nullptr == gradImg_y.data)
	{
		std::cout << "2.image is null!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows != gradImg_x.rows || grayImg.cols != gradImg_x.cols)
	{
		std::cout << "2.sizes don't match!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows != gradImg_y.rows || grayImg.cols != gradImg_y.cols)

{
		std::cout << "2.sizes don't match!\n";

return SUB_IMAGE_MATCH_FAIL;
	}
	
	int row = grayImg.rows - 1;
	int col = grayImg.cols - 1;
	float *px = (float*)gradImg_x.data + gradImg_x.cols + 1;
	float *py = (float*)gradImg_y.data + gradImg_y.cols + 1;
	uchar *px1 = grayImg.data, *px2 = px1 + 2, *px3 = px1 + grayImg.cols, *px4 = px3 + 2, *px5 = px3 + grayImg.cols, *px6 = px5 + 2;
	uchar *py1 = grayImg.data, *py2 = py1 + 1, *py3 = py2 + 1, *py4 = py1 + grayImg.cols * 2, *py5 = py4 + 1, *py6 = py5 + 1;
	int i, j;
	for (i = grayImg.rows - 2;i > 0;i--)
	{
		for (j = grayImg.cols - 2;j > 0;j--)
		{
			*px = *px1 - *px2 + ((*px3) << 2) - ((*px4) << 2) + *px5 - *px6;
			*py = *py1 + ((*py2) << 2) + *py3 - *py4 - ((*py5) << 2) - *py6;

			px++;px1++;px2++;px3++;px4++;px5++;px6++;
			py++;py1++;py2++;py3++;py4++;py5++;py6++;
		}
		px++;px1++;px2++;px3++;px4++;px5++;px6++;
		py++;py1++;py2++;py3++;py4++;py5++;py6++;

		px++;px1++;px2++;px3++;px4++;px5++;px6++;
		py++;py1++;py2++;py3++;py4++;py5++;py6++;
	}
	
	return SUB_IMAGE_MATCH_OK;
}

//3.幅值图和角度图
int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (nullptr == gradImg_x.data || nullptr == gradImg_y.data || nullptr == angleImg.data || nullptr == magImg.data)
	{
		std::cout << "3.image is null!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}

	int i = gradImg_x.rows*gradImg_x.cols;
	float* px = (float*)gradImg_x.data + i - 1;
	float* py = (float*)gradImg_y.data + i - 1;
	float* pa = (float*)angleImg.data + i - 1;
	float* pm = (float*)magImg.data + i - 1;
	float angle;

	for (;i > 0;i--)
	{
		angle = atan2(*py, *px);
		if (angle >= 0)
		{
			*pa = angle * 57.2958f;  // 180/pi = 57.2958
		}
		else
		{
			*pa = angle * 57.2958f + 360;
		}
		*pm = sqrt(*px**px + *py**py);
		px--;py--;pa--;pm--;
	}
	
	return SUB_IMAGE_MATCH_OK;
}

//4.灰度转二值
int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	if (nullptr == grayImg.data || nullptr == binaryImg.data)
	{
		cout << "4.image is null!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows != binaryImg.rows || grayImg.cols != binaryImg.cols)
	{
		cout << "4.sizes don't match!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}

	int i;
	uchar val[256] = { 0 };
	for (i = 50;i < 256;i++)
	{
		val[i] = 255;
	}

	uchar *p = grayImg.data + grayImg.cols * grayImg.rows - 1;
	uchar *q = binaryImg.data + grayImg.cols * grayImg.rows - 1;
	for (i = grayImg.cols * grayImg.rows;i > 0;--i)
	{
		*q = val[*p];
		--p;--q;
	}

	return SUB_IMAGE_MATCH_OK;
}

//5.计算直方图
int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
	if (nullptr == grayImg.data)
	{
		std::cout << "5.image is null!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (nullptr == hist)
	{
		std::cout << "5.hist is null!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}

	int num = grayImg.rows*grayImg.cols;
	int i, j;
	uchar *p = grayImg.data + grayImg.rows*grayImg.cols - 1;

	for (i = 255;i >= 0;i--)//清空直方图
	{
		hist[i] = 0;
	}

	i = num >> 4; //num/16
	j = num - ((num >> 4) << 4);

	for (;j > 0;j--)
	{
		hist[*p]++;
		p--;
	}
	for (;i > 0;i--)
	{
		hist[*p]++; p--;   hist[*p]++; p--;
		hist[*p]++; p--;   hist[*p]++; p--;
		hist[*p]++; p--;   hist[*p]++; p--;
		hist[*p]++; p--;   hist[*p]++; p--;
		hist[*p]++; p--;   hist[*p]++; p--;
		hist[*p]++; p--;   hist[*p]++; p--;
		hist[*p]++; p--;   hist[*p]++; p--;
		hist[*p]++; p--;   hist[*p]++; p--;
	}

	return SUB_IMAGE_MATCH_OK;
}

//6.利用亮度进行子图匹配
int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (nullptr == grayImg.data || nullptr == subImg.data)
	{
		std::cout << "6.image is null!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows < subImg.rows || grayImg.cols < subImg.cols)
	{
		std::cout << "6.subImg is lager than grayImg!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (nullptr == x || nullptr == y)
	{
		std::cout << "6.x or y dont exist!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}

	int row_scan = grayImg.rows - subImg.rows;
	int col_scan = grayImg.cols - subImg.cols;
	int sub_row_scan = subImg.rows - 1;
	int sub_col_scan = subImg.cols - 1;
	uchar *p;//大图计算用指针
	uchar *q;//子图计算用指针
	int i, j, m;//循环变量
	int shang, yushu;//最内层循环使用
	int temp_gray_i;
	int minx = 0, miny = 0;//亮度差之和最小子图左上角坐标
	int sum_min = INT_MAX;//亮度差之和最小值
	int sum = 0;//亮度差之和
	int sub;//亮度差

	for (i = row_scan;i >= 0;i--)
	{
		temp_gray_i = i*grayImg.cols;
		for (j = col_scan;j >= 0;j--)
		{
			sum = 0;
			//内层循环
			for (m = sub_row_scan;m >= 0;m--)
			{
				p = grayImg.data + temp_gray_i + j + m*grayImg.cols;//行首
				q = subImg.data + m*subImg.cols;
				shang = subImg.cols >> 3;
				yushu = subImg.cols - ((subImg.cols >> 3) << 3);
				for (p += sub_col_scan, q += sub_col_scan;yushu > 0;yushu--)
				{
					sub = *p - *q;
					if (sub > 0) sum += sub;
					else sum -= sub;
					p--;q--;
				}
				for (;shang > 0;shang--)
				{
					sub = *p - *q;   if (sub > 0) sum += sub;   else sum -= sub; p--; q--;//处理一个像素
					sub = *p - *q;   if (sub > 0) sum += sub;   else sum -= sub; p--; q--;
					sub = *p - *q;   if (sub > 0) sum += sub;   else sum -= sub; p--; q--;
					sub = *p - *q;   if (sub > 0) sum += sub;   else sum -= sub; p--; q--;
					sub = *p - *q;   if (sub > 0) sum += sub;   else sum -= sub; p--; q--;
					sub = *p - *q;   if (sub > 0) sum += sub;   else sum -= sub; p--; q--;
					sub = *p - *q;   if (sub > 0) sum += sub;   else sum -= sub; p--; q--;
					sub = *p - *q;   if (sub > 0) sum += sub;   else sum -= sub; p--; q--;
				}
			}
			if (sum < sum_min)
			{
				sum_min = sum;
				minx = j;
				miny = i;
			}
		}
	}

	*x = minx;
	*y = miny;
	return SUB_IMAGE_MATCH_OK;
}

//7.利用色彩进行子图匹配
int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
	if (nullptr == colorImg.data || nullptr == subImg.data)
	{
		std::cout << "7.image is null!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (colorImg.rows < subImg.rows || colorImg.cols < subImg.cols)
	{
		std::cout << "7.subImage is lager than colorImg!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (nullptr == x || nullptr == y)
	{
		std::cout << "7.x or y dont exist!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}

	int row_scan = colorImg.rows - subImg.rows;
	int col_scan = colorImg.cols - subImg.cols;
	int sub_row_scan = subImg.rows - 1;
	int sub_col_scan = subImg.cols * 3 - 1;
	int color_col = colorImg.cols * 3;
	int sub_col = subImg.cols * 3;
	int i, j, m;//循环变量
	int k;
	int shang, yushu;
	int temp_color_i;
	uchar *p;//计算用指针
	uchar *q;//子图计算用指针
	int temp;

	int minx = 0, miny = 0;//色彩差之和最小子图左上角坐标
	int msum = INT_MAX;//色彩差之和最小值
	int sum = 0;//色彩差之和
	char sub;//色彩差
	char sub1, sub2, sub3, sub4;

	for (i = row_scan;i >= 0;i--)
	{
		temp_color_i = i*color_col;
		for (j = col_scan;j >= 0;j--)
		{
			sum = 0;
			k = 3 * j;
			for (m = sub_row_scan;m >= 0;m--)
			{
				p = colorImg.data + temp_color_i + k + m * color_col;//行首
				q = subImg.data + m * sub_col;//行首
				shang = sub_col >> 2;
				yushu = sub_col - ((sub_col >> 2) << 2);
				for (;yushu > 0;yushu--)
				{
					sub = *p - *q;
					if (sub > 0)
						sum += sub;
					else
						sum -= sub;
					p++;q++;
				}
				for (;shang > 0;shang--)
				{
					sub1 = *p;p++;   sub2 = *p;p++;   sub3 = *p;p++;  sub4 = *p;p++;  
					sub1 -= *q;q++;  sub2 -= *q;q++;  sub3 -= *q;q++; sub4 -= *q;q++; 
					if (sub1 > 0) sum += sub1;else sum -= sub1;
					if (sub2 > 0) sum += sub2;else sum -= sub2;
					if (sub3 > 0) sum += sub3;else sum -= sub3;
					if (sub4 > 0) sum += sub4;else sum -= sub4;
				}
			}
			if (sum < msum)
			{
				msum = sum;
				minx = j;
				miny = i;
			}
		}
	}

	*x = minx;
	*y = miny;
	return SUB_IMAGE_MATCH_OK;
}

//8.利用亮度相关性进行子图匹配
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (nullptr == grayImg.data || nullptr == subImg.data)
	{
		std::cout << "8.image is null!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows < subImg.rows || grayImg.cols < subImg.cols)
	{
		std::cout << "8.subImg is lager than grayImg!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (nullptr == x || nullptr == y)
	{
		std::cout << "8.x or y dont exist!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}

	int row_scan = grayImg.rows - subImg.rows;
	int col_scan = grayImg.cols - subImg.cols;
	int sub_row_scan = subImg.rows - 1;
	int sub_col_scan = subImg.cols - 1;
	int temp_gray = 0, temp_sub = 0;//索引
	int temp_gray_i, temp_gray_m;//行索引
	int i, j, m;//ij为大图循环变量，mn为子图循环变量
	long long gray_sub = 0, gray_gray = 0, sub_sub = 0;//s*t,s*s,t*t
	int s_gray1, s_gray2, s_gray3, s_gray4;
	int t_sub1, t_sub2, t_sub3, t_sub4;
	float relation = 0, relation_max = 0;//相关性及其最大值
	int x_max = 0, y_max = 0;
	uchar *p, *q;
	int shang, yushu;

	for (i = row_scan;i >= 0;i--)
	{
		temp_gray_i = i*grayImg.cols;
		for (j = col_scan;j >= 0;j--)
		{
			//子图循环
			gray_sub = 0;gray_gray = 0;sub_sub = 0;
			for (m = sub_row_scan;m >= 0;m--)
			{
				p = grayImg.data + temp_gray_i + j + m*grayImg.cols;//行首
				q = subImg.data + m*subImg.cols;//行首
				shang = subImg.cols >> 2;
				yushu = subImg.cols - ((subImg.cols >> 2) << 2);
				for (;yushu > 0;yushu--)
				{
					s_gray1 = *p; p++;
					t_sub1 = *q; q++;
					gray_sub += s_gray1 * t_sub1;
					gray_gray += s_gray1 * s_gray1;
					sub_sub += t_sub1 * t_sub1;
				}
				for (;shang > 0;shang--)
				{
					s_gray1 = *p;p++; s_gray2 = *p;p++; s_gray3 = *p;p++; s_gray4 = *p;p++;
					t_sub1 = *q;q++;  t_sub2 = *q;q++;  t_sub3 = *q;q++;  t_sub4 = *q;q++;
					gray_sub += s_gray1 * t_sub1 + s_gray2 * t_sub2 + s_gray3 * t_sub3 + s_gray4 * t_sub4;
					gray_gray += s_gray1 * s_gray1 + s_gray2 * s_gray2 + s_gray3 * s_gray3 + s_gray4 * s_gray4;
					sub_sub += t_sub1 * t_sub1 + t_sub2 * t_sub2 + t_sub3 * t_sub3 + t_sub4 * t_sub4;
				}
			}
			relation = (float)gray_sub / sqrt(gray_gray*sub_sub);
			if (relation > relation_max)
			{
				x_max = j;
				y_max = i;
				relation_max = relation;
			}
		}
	}

	*x = x_max;
	*y = y_max;
	return SUB_IMAGE_MATCH_OK;
}

//9.利用角度值进行子图匹配
int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (nullptr == grayImg.data || nullptr == subImg.data)
	{
		std::cout << "9.image is null!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows < subImg.rows || grayImg.cols < subImg.cols)
	{
		std::cout << "9.subImg is lager than grayImg!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (nullptr == x || nullptr == y)
	{
		std::cout << "9.x or y dont exist!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}

	Mat grayImg_angle(grayImg.rows, grayImg.cols, CV_8UC1);
	Mat subImg_angle(subImg.rows, subImg.cols, CV_8UC1);
	int i, j, m, n;//循环变量
	int grad_x, grad_y;//梯度
	int row_1, row_2, row_3;
	int col_1, col_3;
	uchar *data;
	int anglesum, anglesum_min = INT_MAX;//角度差之和及其最小值
	int x_min, y_min;//坐标
	int scan_rows = grayImg.rows - subImg.rows + 1;//大图扫描行数
	int scan_cols = grayImg.cols - subImg.cols + 1;//大图扫描列数
	int angle;
	int sub;
	uchar *p, *q;
	int shang, yushu;
	int temp_gray_i;

	//灰度图求角度
	data = grayImg.data;  
	for (i = 1;i < grayImg.rows - 1;i++)
	{
		row_1 = (i - 1)*grayImg.cols;
		row_2 = i*grayImg.cols;
		row_3 = (i + 1)*grayImg.cols;
		for (j = 1;j < grayImg.cols - 1;j++)
		{
			col_1 = j - 1;
			col_3 = j + 1;
			grad_x = data[row_1 + col_1] + data[row_2 + col_1] << 1 + data[row_3 + col_1] - data[row_1 + col_3] - data[row_2 + col_3] << 1 - data[row_3 + col_3];
			grad_y = data[row_1 + col_1] + data[row_1 + j] << 1 + data[row_1 + col_3] - data[row_3 + col_1] - data[row_3 + j] << 1 - data[row_3 + col_3];
			angle = atan2(grad_y, grad_x);
			if (angle >= 0)
			{
				grayImg_angle.data[row_2 + j] = angle * 57.2958f;  // 180/pi = 57.2958
			}
			else
			{
				grayImg_angle.data[row_2 + j] = angle * 57.2958f + 360;
			}   
		}
	}

	//模板子图求角度
	data = subImg.data;  
	for (i = 1;i < subImg.rows - 1;i++)
	{
		row_1 = (i - 1)*subImg.cols;
		row_2 = i*subImg.cols;
		row_3 = (i + 1)*subImg.cols;
		for (j = 1;j < subImg.cols - 1;j++)
		{
			col_1 = j - 1;
			col_3 = j + 1;
			grad_x = data[row_1 + col_1] + data[row_2 + col_1] << 1 + data[row_3 + col_1] - data[row_1 + col_3] - data[row_2 + col_3] << 1 - data[row_3 + col_3];
			grad_y = data[row_1 + col_1] + data[row_1 + j] << 1 + data[row_1 + col_3] - data[row_3 + col_1] - data[row_3 + j] << 1 - data[row_3 + col_3];
			angle = atan2(grad_y, grad_x);
			if (angle >= 0)
			{
				subImg_angle.data[row_2 + j] = angle * 57.2958f;  // 180/pi = 57.2958
			}
			else
			{
				subImg_angle.data[row_2 + j] = angle * 57.2958f + 360;
			}
		}
	}

	//匹配求坐标
	for (i = 0;i < scan_rows;i++)
	{
		temp_gray_i = i * grayImg_angle.cols;
		for (j = 0;j < scan_cols;j++)
		{
			//子图循环
			anglesum = 0;
			for (m = 0;m < subImg_angle.rows;m++)
			{
				p = grayImg_angle.data + temp_gray_i + j + m * grayImg_angle.cols;//行首
				q = subImg_angle.data + m * subImg_angle.cols;//行首
				//row_2 = row_1 + j + m*grayImg_angle.cols;//temp_gray
				//row_3 = m*subImg_angle.cols;//temp_sub
				shang = subImg_angle.cols >> 3;
				yushu = subImg_angle.cols - ((subImg_angle.cols >> 3) << 3);
				for (;yushu > 0;yushu--)
				{
					anglesum += ((*p - *q) + 360) % 180;
					p++;q++;
				}
				for (;shang > 0;shang--)
				{
					anglesum += ((*p - *q) + 360) % 180;  p++;q++;
					anglesum += ((*p - *q) + 360) % 180;  p++;q++;
					anglesum += ((*p - *q) + 360) % 180;  p++;q++;
					anglesum += ((*p - *q) + 360) % 180;  p++;q++;
					anglesum += ((*p - *q) + 360) % 180;  p++;q++;
					anglesum += ((*p - *q) + 360) % 180;  p++;q++;
					anglesum += ((*p - *q) + 360) % 180;  p++;q++;
					anglesum += ((*p - *q) + 360) % 180;  p++;q++;
				}
			}
			if (anglesum < anglesum_min)
			{
				x_min = j;
				y_min = i;
				anglesum_min = anglesum;
			}
		}
	}

	*x = x_min;
	*y = y_min;
	return SUB_IMAGE_MATCH_OK;
}

//10.利用幅值进行子图匹配
int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (nullptr == grayImg.data || nullptr == subImg.data)
	{
		std::cout << "10.image is null!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows < subImg.rows || grayImg.cols < subImg.cols)
	{
		std::cout << "10.subImg is lager than grayImg!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (nullptr == x || nullptr == y)
	{
		std::cout << "10.x or y dont exist!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}

	Mat grayImg_mag(grayImg.rows, grayImg.cols, CV_8UC1);//大图幅值图
	Mat subImg_mag(subImg.rows, subImg.cols, CV_8UC1);//子图幅值图
	int i, j, m;//循环变量
	int grad_x, grad_y;//梯度
	int row_1, row_2, row_3;
	int col_1, col_3;
	uchar *data;
	int magsum, magsum_min = INT_MAX;//幅值差之和及其最小值
	int x_min, y_min;//坐标
	int scan_rows = grayImg.rows - subImg.rows + 1;//大图扫描行数
	int scan_cols = grayImg.cols - subImg.cols + 1;//大图扫描列数
	int sub;
	int temp_gray_i;
	uchar *p, *q;
	int shang, yushu;

	//灰度图求幅值
	data = grayImg.data;
	for (i = grayImg.rows - 2;i > 0;i--)
	{
		row_1 = (i - 1)*grayImg.cols;
		row_2 = i*grayImg.cols;
		row_3 = (i + 1)*grayImg.cols;
		for (j = grayImg.cols - 2;j > 0;j--)
		{
			col_1 = j - 1;
			col_3 = j + 1;
			grad_x = data[row_1 + col_1] + data[row_2 + col_1] << 1 + data[row_3 + col_1] - data[row_1 + col_3] - data[row_2 + col_3] << 1 - data[row_3 + col_3];
			grad_y = data[row_1 + col_1] + data[row_1 + j] << 1 + data[row_1 + col_3] - data[row_3 + col_1] - data[row_3 + j] << 1 - data[row_3 + col_3];
			grayImg_mag.data[row_2 + j] = sqrt(grad_x*grad_x + grad_y*grad_y);    
		}
	}

	//模板子图求幅值
	data = subImg.data;
	for (i = 1;i < subImg.rows - 1;i++)
	{
		row_1 = (i - 1)*subImg.cols;
		row_2 = i*subImg.cols;
		row_3 = (i + 1)*subImg.cols;
		for (j = 1;j < subImg.cols - 1;j++)
		{
			col_1 = j - 1;
			col_3 = j + 1;
			grad_x = data[row_1 + col_1] + data[row_2 + col_1] << 1 + data[row_3 + col_1] - data[row_1 + col_3] - data[row_2 + col_3] << 1 - data[row_3 + col_3];
			grad_y = data[row_1 + col_1] + data[row_1 + j] << 1 + data[row_1 + col_3] - data[row_3 + col_1] - data[row_3 + j] << 1 - data[row_3 + col_3];
			subImg_mag.data[row_2 + j] = sqrt(grad_x*grad_x + grad_y*grad_y);
		}
	}

	//匹配求坐标
	for (i = 0;i < scan_rows;i++)
	{
		temp_gray_i = i*grayImg_mag.cols;
		for (j = 0;j < scan_cols;j++)
		{
			//子图循环
			magsum = 0;
			for (m = 0;m < subImg_mag.rows;m++)
			{
				p = grayImg_mag.data + temp_gray_i + j + m * grayImg_mag.cols;//行首
				q = subImg_mag.data + m * subImg_mag.cols;//子图行首
				shang = subImg_mag.cols >> 2;
				yushu = subImg_mag.cols - ((subImg_mag.cols >> 2) << 2);
				for (;yushu > 0;yushu--)
				{
					sub = *p - *q; p++; q++;
					if (sub > 0)
						magsum += sub;
					else
						magsum -= sub;
				}
				for (;shang > 0;shang--)
				{
					sub = *p - *q;  if (sub > 0) magsum += sub; else magsum -= sub;        p++; q++;
					sub = *p - *q;  if (sub > 0) magsum += sub; else magsum -= sub;        p++; q++;
					sub = *p - *q;  if (sub > 0) magsum += sub; else magsum -= sub;        p++; q++;
					sub = *p - *q;  if (sub > 0) magsum += sub; else magsum -= sub;        p++; q++;
				}
			}
			if (magsum < magsum_min)
			{
				x_min = j;
				y_min = i;
				magsum_min = magsum;
			}
		}
	}

	*x = x_min;
	*y = y_min;
	return SUB_IMAGE_MATCH_OK;
}

//11.利用直方图进行子图匹配
int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (nullptr == grayImg.data || nullptr == subImg.data)
	{
		std::cout << "11.image is null!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (grayImg.rows < subImg.rows || grayImg.cols < subImg.cols)
	{
		std::cout << "11.subImg is larger than grayImg!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}
	if (nullptr == x || nullptr == y)
	{
		std::cout << "11.x or y dont exist!\n";
		return SUB_IMAGE_MATCH_FAIL;
	}

	int gray_hist[256] = { 0 };
	int sub_hist[256] = { 0 };
	int i, j, m, n;//循环变量
	int scan_rows = grayImg.rows - subImg.rows;
	int scan_cols = grayImg.cols - subImg.cols;
	int sum, sum_min = INT_MAX;//差之和 差之和最小值
	int x_min, y_min;//坐标
	int sub_row_1 = subImg.rows - 1;
	int sub_col_1 = subImg.cols - 1;
	int temp_row_i;
	int shang, yushu;
	uchar *p;//指向计算的像素

	//子图直方图
	for (i = subImg.rows*subImg.cols - 1;i >= 0;i--)
	{
		sub_hist[*(subImg.data + i)]++;
	}

	//匹配外层循环
	for (i = scan_rows;i >= 0;i--)
	{
		temp_row_i = i*grayImg.cols;
		for (j = scan_cols;j >= 0;j--)
		{
			//前一个匹配位置直方图置零
			for (m = 255;m >= 0;m--)
			{
				gray_hist[m] = 0;
			}
			//计算匹配位置直方图
			for (m = sub_row_1;m >= 0;m--)
			{
				p = grayImg.data + temp_row_i + j + m * grayImg.cols;//行首
				shang = subImg.cols >> 3;
				yushu = subImg.cols - ((subImg.cols >> 3) << 3);
				/*for (n = sub_col_1;n >= 0;n--)
				{
					gray_hist[*p]++;p++;
				}*/
				for (;shang > 0;shang--)
				{
					gray_hist[*p]++;p++;
					gray_hist[*p]++;p++;
					gray_hist[*p]++;p++;
					gray_hist[*p]++;p++;
					gray_hist[*p]++;p++;
					gray_hist[*p]++;p++;
					gray_hist[*p]++;p++;
					gray_hist[*p]++;p++;
				}
				for (;yushu > 0;yushu--)
				{
					gray_hist[*p]++;
					p++;
				}
			}
			//计算直方图之差
			sum = 0;
			for (m = 255;m >= 0;m--)
			{
				sum += abs(gray_hist[m] - sub_hist[m]);
			}
			if (sum < sum_min)
			{
				x_min = j;
				y_min = i;
				sum_min = sum;
			}
		}
	}

	*x = x_min;
	*y = y_min;
	return SUB_IMAGE_MATCH_OK;
}






