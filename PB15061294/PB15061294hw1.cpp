#include "SubImageMatch.h"
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
	if (NULL == bgrImg.data || NULL == grayImg.data)
	{
		cout << "the image is NULL" << endl;
		return 0;
	}
	if (bgrImg.channels()!=3||grayImg.channels()!=1 )
	{
		cout << "the formats of the pictures are not right" << endl;
		return 0;
	}
	int height = bgrImg.rows;
	int width = bgrImg.cols;
	for (int i = 0;i < height;i++)
	{
		int length = i;
		for (int j = 0;j < width;j++)
		{
			int blue = bgrImg.data[3 * (length*width + j) + 0];
			int green = bgrImg.data[3 * (length*width + j) + 1];
			int red = bgrImg.data[3 * (length*width + j) + 2];
			int grayval = blue*0.114f + green*0.587f + red*0.299f;
			grayImg.data[length*width + j] = grayval;
		}
	}

	return 0;
}
int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
	if (NULL == grayImg.data || NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "the image is NULL" << endl;
		return 0;
	}
	if (grayImg.channels() != 1 || gradImg_x.channels() != 1 || gradImg_y.channels() != 1)
	{
		cout << " the format is not right" << endl;
		return 0;
	}
	gradImg_x.setTo(0);
	gradImg_y.setTo(0);
	int height = grayImg.rows;
	int width = grayImg.cols;
	for (int i = 1;i < height - 1;i++)
	{
		int length = i;
		for (int j = 1;j < width - 1;j++)
		{
			
			int grad_x = grayImg.data[(length - 1)*width + j + 1] + 2 * grayImg.data[length*width + j + 1] + grayImg.data[(length + 1)*width + j + 1]
				- grayImg.data[(length - 1)*width + j - 1] - 2 * grayImg.data[length*width + j - 1] - grayImg.data[(length + 1)*width + j - 1];
			((float*)gradImg_x.data)[i*width + j] = grad_x;
		}
	}
	for (int k = 1;k < height - 1;k++)
	{
		int length = k;
		for (int l = 1;l < width - 1;l++)
		{
			int grad_y = grayImg.data[(length + 1)*width + l - 1] + 2 * grayImg.data[(length + 1)*width + l] + grayImg.data[(length + 1)*width + l + 1]
				- grayImg.data[(length - 1)*width + l - 1] - 2 * grayImg.data[(length - 1)*width + l] - grayImg.data[(length - 1)*width + l + 1];
			((float*)gradImg_y.data)[length*width + l] = grad_y;
		}
	}
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	if (NULL == gradImg_x.data || NULL == gradImg_y.data || NULL == angleImg.data || NULL == magImg.data)
	{
		cout << "the image is NULL" << endl;
		return 0;
	}
	if (angleImg.channels() != 1 || gradImg_x.channels() != 1 || gradImg_y.channels() != 1||magImg.channels()!=1)
	{
		cout << " the format is not right" << endl;
		return 0;
	}
	int height = gradImg_x.rows;
	int width = gradImg_y.cols;
	angleImg.setTo(0);
	magImg.setTo(0);
	for (int i = 1;i < height - 1;i++)
	{
		for (int j = 1;j < width - 1;j++)
		{
			float grad_x = ((float*)gradImg_x.data)[i*width + j];
			float grad_y = ((float*)gradImg_y.data)[i*width + j];
			float angle = 0;
			if (grad_y < 0) grad_y = -grad_y;
			if (grad_x < 0) grad_x = -grad_x;
			float u = grad_y / grad_x;
			if (u <= 1)
			{
				angle = 90 * u / (1 + u) + 16 * u*u - 16 * u;
			}
			else if (u > 1)
			{
				float v = 1 / u;
				angle = 90 / (1 + v) + 16 * v - 16 * v*v;
			}
			float result = angle;
			if (grad_y < 0)
			{
				if (grad_x < 0)
					result = angle - 180;
				else result = 0 - angle;
			}
			else if(grad_y>=0)
			{
				if (grad_x < 0)
					result = 180 - angle;
			}
			((float*)angleImg.data)[i*width + j] = result;
		}
	}
	for (int i = 1;i < height - 1;i++)
	{
		for (int j = 1;j < width - 1;j++)
		{
			float grad_x = ((float*)gradImg_x.data)[i*width + j];
			float grad_y = ((float*)gradImg_y.data)[i*width + j];
			float square = grad_x*grad_x + grad_y*grad_y;
			int t = *(int*)&square;
			t -= 0x3f800000;
			t >>= 1;
			t += 0x3f800000;
			square = *(float*)&t;
			float magnitude =square;
			((float*)magImg.data)[i*width + j] = magnitude;
		}
	}
	
}
int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
	if (NULL == grayImg.data || NULL == binaryImg.data)
	{
		cout << "the image is NULL" << endl;
		return 0;
	}
	if (grayImg.channels() != 1 || binaryImg.channels() != 1)
	{
		cout << "the format is incorrect" << endl;
		return 0;
	}
	if (th < 0 || th>255)
	{
		cout << "extension of the th value" << endl;
		return 0;
	}
	int height = grayImg.rows;
	int width = grayImg.cols;
	int size = height*width;
	for(int number=0;number<size;number++)
	{
		int grayval = grayImg.data[number];
			if (grayval > th)
			{
				binaryImg.data[number] = 255;
			}
			else
			{
				binaryImg.data[number] = 0;
			}
			
		}
	
}
int ustc_CalcHist(Mat grayImg, int *hist, int hist_len)
{
	if (NULL == grayImg.data || NULL == hist)
	{
		cout << "image is NULL" << endl;
		return 0;
	}
	if (hist_len < 256)
	{
		cout << "it can not structure a complete array of pixvalues" << endl;
		return 0;
	}
	int height = grayImg.rows;
	int width = grayImg.cols;
	for (int i = 0;i < hist_len;i++)
	{
		hist[i] = 0;
	}
	for (int i = 0;i < height;i++)
	{
		for (int j = 0;j < width;j++)
		{
			int pixval = grayImg.data[i*width + j];
			hist[pixval]++;
		}
	}
}
int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int *x, int *y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL" << endl;
		return 0;
	}
	if (grayImg.channels() != 1 || subImg.channels() != 1)
	{
		cout << "the format is incorrect" << endl;
		return 0;
	}
	int height = grayImg.rows;
	int width = grayImg.cols;
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;
	int graysize = height*width;
	int subsize = sub_height*sub_width;
	if (graysize < subsize || height < sub_height || width < sub_width)
	{
		cout << "the subImg can not match the measurement of the bigImg" << endl;
		return 0;
	}
	int match_x;
	int match_y;
	int match_difference = INT32_MAX;
	for (int i = 0;i < height - sub_height;i++)
	{
		for (int j = 0;j < width - sub_width;j++)
		{
			int total_difference = 0;
			for (int x = 0;x < sub_height;x++)
			{
				for (int y = 0;y < sub_width;y++)
				{
					int row_index = i + x;
					int col_index = j + y;
					int bigImg_pix = grayImg.data[row_index*width + col_index];
					int template_pix = subImg.data[x*sub_width + y];
					if (bigImg_pix > template_pix)
						total_difference += bigImg_pix - template_pix;
					else total_difference += template_pix - bigImg_pix;
				}
			}
			if (total_difference < match_difference)
			{
				match_x = i;
				match_y = j;
				match_difference = total_difference;
			}
		}
	}
	printf("the vexpoint position is (%d,%d)", match_x, match_y);
	x = &match_x;
	y = &match_y;

	return 0;
}
int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int *x, int *y)
{
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << " the image is NULL" << endl;
		return 0;
	}
	if (colorImg.channels() != 3 || subImg.channels() != 3)
	{
		cout << "the format is incorrect" << endl;
		return 0;
	}
	int height = colorImg.rows;
	int width = colorImg.cols;
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;
	int graysize = height*width;
	int subsize = sub_height*sub_width;
	if (graysize < subsize || height < sub_height || width < sub_width)
	{
		cout << "the subImg can not match the measurement of the bigImg" << endl;
		return 0;
	}
	int match_x;
	int match_y;
	int match_difference = INT32_MAX;
	for (int i = 0;i < height - sub_height;i++)
	{
		for (int j = 0;j < width - sub_width;j++)
		{
			int total_difference = 0;
			for (int x = 0;x < sub_height;x++)
			{
				for (int y = 0;y < sub_width;y++)
				{
					int row_index = i + x;
					int col_index = j + y;
					int bigImg_sequence = 3 * (row_index*width + col_index);
					int subImg_sequence = 3 * (x*sub_width + y);
					int bigImgblue = colorImg.data[bigImg_sequence + 0];
					int subImgblue = subImg.data[subImg_sequence + 0];
					if (bigImgblue > subImgblue)
						total_difference += bigImgblue - subImgblue;
					else total_difference += subImgblue - bigImgblue;
					int bigImggreen = colorImg.data[bigImg_sequence+ 1];
					int subImggreen = subImg.data[subImg_sequence + 1];
					if (bigImggreen > subImggreen)
						total_difference += bigImggreen - subImggreen;
					else total_difference += subImggreen - bigImggreen;
					int bigImgred = colorImg.data[bigImg_sequence + 2];
					int subImgred = subImg.data[subImg_sequence + 2];
					if (bigImgred > subImgred)
						total_difference += bigImgred - subImgred;
					else total_difference += subImgred - bigImgred;
				}
			}
			if (total_difference < match_difference)
			{
				match_difference = total_difference;
				match_x = i;
				match_y = j;
			}
		}
	}

	printf("the vexpoint position is (%d,%d)", match_x, match_y);
	x = &match_x;
	y = &match_y;
	return 0;
}

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int *x, int *y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "the image in NULL" << endl;
		return 0;
	}
	if (grayImg.channels() != 1 || subImg.channels() != 1)
	{
		cout << " the format is incorrect" << endl;
		return 0;
	}
	int height = grayImg.rows;
	int width = grayImg.cols;
	float relation = 0;
	float maxrelation = 0;
	int match_x = 0;
	int match_y = 0;
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;
	int graysize = height*width;
	int subsize = sub_height*sub_width;
	if (graysize < subsize || height < sub_height || width < sub_width)
	{
		cout << "the subImg can not match the measurement of the bigImg" << endl;
		return 0;
	}
	for (int i = 0;i < height - sub_height;i++)
	{
		for (int j = 0;j < width - sub_width;j++)
		{
			int total_big = 0;
			int total_sub = 0;
			int total_bigsub = 0;
			for (int m = 0;m < sub_height;m++)
			{
				for (int n = 0;n < sub_width;n++)
				{
					int row_index = i + m;
					int col_index = j + n;
					int bigImgpix = grayImg.data[row_index*width + col_index];
					int subImgpix = subImg.data[m*sub_width + n];
					total_big += bigImgpix*bigImgpix;
					total_sub += subImgpix*subImgpix;
					total_bigsub += bigImgpix*subImgpix;
				}
			}
			float c1 = float(total_bigsub) / total_big;
			float c2 = float(total_bigsub) / total_sub;
			relation = c1*c2;
			if (relation > maxrelation)
			{
				maxrelation = relation;
				match_x = i;
				match_y = j;
			}
			
		}
	}


	printf("the vexpoint position is (%d,%d)", match_x, match_y);
	x = &match_x;
	y = &match_y;

	return 0;
}
int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int *x, int *y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "the angleimage is NULL" << endl;
		return 0;
	}
	if (grayImg.channels() != 1 || subImg.channels() != 1)
	{
		cout << "the format is incorrect" << endl;
		return 0;
	}
	int height = grayImg.rows;
	int width = grayImg.cols;
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;
	int graysize = height*width;
	int subsize = sub_height*sub_width;
	if (graysize < subsize || height < sub_height || width < sub_width)
	{
		cout << "the subImg can not match the measurement of the bigImg" << endl;
		return 0;
	}
	Mat graygradImg_x(height, width, CV_32FC1);
	graygradImg_x.setTo(0);
	Mat graygradImg_y(height, width, CV_32FC1);
	graygradImg_y.setTo(0);
	ustc_CalcGrad(grayImg, graygradImg_x, graygradImg_y);
	Mat grayangleImg(height, width, CV_32FC1);
	grayangleImg.setTo(0);
	Mat graymagnitudeImg(height, width, CV_32FC1);
	graymagnitudeImg.setTo(0);
	ustc_CalcAngleMag(graygradImg_x, graygradImg_y, grayangleImg, graymagnitudeImg);
	Mat grayangleImg_8U(height, width, CV_8UC1);
	for (int i = 0;i < height;i++)
	{
		for (int j = 0;j < width;j++)
		{
			int angle = ((float*)grayangleImg.data)[i*width + j];
			angle += 180;
			angle /= 2;
			grayangleImg_8U.data[i*width + j] = angle;
		}
	}//将角度由-180-180之间的浮点数转化为0-180之间的整数
	Mat subgradImg_x(sub_height, sub_width, CV_32FC1);
	subgradImg_x.setTo(0);
	Mat subgradImg_y(sub_height, sub_width, CV_32FC1);
	subgradImg_y.setTo(0);
	ustc_CalcGrad(subImg, subgradImg_x, subgradImg_y);
	Mat subangleImg(sub_height, sub_width, CV_32FC1);
	subangleImg.setTo(0);
	Mat submagnitudeImg(sub_height, sub_width, CV_32FC1);
	submagnitudeImg.setTo(0);
	ustc_CalcAngleMag(subgradImg_x, subgradImg_y, subangleImg, submagnitudeImg);
	Mat subangleImg_8U(sub_height, sub_width, CV_8UC1);
	for (int i = 0;i < sub_height;i++)
	{
		for (int j = 0;j < sub_width;j++)
		{
			int angle = ((float*)subangleImg.data)[i*sub_width + j];
			angle += 180;
			angle /= 2;
			subangleImg_8U.data[i*sub_width + j] = angle;
		}
	}//将角度由-180-180之间的浮点数转化为0-180之间的整数
	int match_x;
	int match_y;
	int match_difference = INT16_MAX;
	
	for (int i = 0;i < height - sub_height;i++)
	{
		for (int j =0;j < width - sub_width;j++)
		{
			int total_difference = 0;
			for (int x = 0;x < sub_height;x++)
			{
				for (int y = 0;y < sub_width;y++)
				{
					int row_index = i + x;
					int col_index = j + y;
					int bigImgangle = grayangleImg_8U.data[row_index*width + col_index];
					int subImgangle = subangleImg_8U.data[x*sub_width + y];
					int difference = bigImgangle - subImgangle;
					if (difference < 0) difference = 0 - difference;
					if (difference > 90) difference = 180 - difference;
					total_difference += difference;
				}
			}//经过线性转换之后的角度之差应该在0-90之间（因为-180-180之间的角度之差在0-180之间）
			if (total_difference<match_difference)
			{
				match_difference = total_difference;
				match_x = i;
				match_y = j;
				
			}
			
		}
	}//find the position of the matched vexpoint
    
	
	printf("the vexpoint position is (%d,%d)", match_x, match_y);
	x = &match_x;
	y = &match_y;
	return 0;
}
int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int *x, int *y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "the angleimage is NULL" << endl;
		return 0;
	}
	if (grayImg.channels() != 1 || subImg.channels() != 1)
	{
		cout << "the format is incorrect" << endl;
		return 0;
	}
	int height = grayImg.rows;
	int width = grayImg.cols;
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;
	int graysize = height*width;
	int subsize = sub_height*sub_width;
	if (graysize < subsize || height < sub_height || width < sub_width)
	{
		cout << "the subImg can not match the measurement of the bigImg" << endl;
		return 0;
	}
	Mat graygradImg_x(height, width, CV_32FC1);
	graygradImg_x.setTo(0);
	Mat graygradImg_y(height, width, CV_32FC1);
	graygradImg_y.setTo(0);
	ustc_CalcGrad(grayImg, graygradImg_x, graygradImg_y);
	Mat grayangleImg(height, width, CV_32FC1);
	grayangleImg.setTo(0);
	Mat graymagnitudeImg(height, width, CV_32FC1);
	graymagnitudeImg.setTo(0);
	ustc_CalcAngleMag(graygradImg_x, graygradImg_y, grayangleImg, graymagnitudeImg);
	Mat subgradImg_x(sub_height, sub_width, CV_32FC1);
	subgradImg_x.setTo(0);
	Mat subgradImg_y(sub_height, sub_width, CV_32FC1);
	subgradImg_y.setTo(0);
	ustc_CalcGrad(subImg, subgradImg_x, subgradImg_y);
	Mat subangleImg(sub_height, sub_width, CV_32FC1);
	subangleImg.setTo(0);
	Mat submagnitudeImg(sub_height, sub_width, CV_32FC1);
	submagnitudeImg.setTo(0);
	ustc_CalcAngleMag(subgradImg_x, subgradImg_y, subangleImg, submagnitudeImg);
	int match_x;
	int match_y;
	float match_difference = 0;
	for (int x = 0;x < sub_height;x++)
	{
		for (int y = 0;y < sub_width;y++)
		{
			float bigImgmagnitude = ((float*)graymagnitudeImg.data)[x*width + y];
			float subImgmagnitude = ((float*)submagnitudeImg.data)[x*sub_width + y];
			match_difference += fabs(bigImgmagnitude - subImgmagnitude);
		}
	}
	for (int i = 0;i < height - sub_height;i++)
	{
		for (int j = 0;j < width - sub_width;j++)
		{
			float total_difference = 0;
			for (int x = 1;x < sub_height - 1;x++)
			{
				for (int y = 1;y < sub_width - 1;y++)
				{
					int row_index = i + x;
					int col_index = j + y;
					float bigImgmagnitude = ((float*)graymagnitudeImg.data)[row_index*width + col_index];
					float subImgmagnitude = ((float*)submagnitudeImg.data)[x*sub_width + y];
					if (bigImgmagnitude > subImgmagnitude)
						total_difference += bigImgmagnitude - subImgmagnitude;
					else total_difference += subImgmagnitude - bigImgmagnitude;

				}
			}
			if (total_difference < match_difference)
			{
				match_x = i;
				match_y = j;
				match_difference = total_difference;
			}
		}
	}

	printf("the vexpoint position is (%d,%d)", match_x, match_y);
	x = &match_x;
	y = &match_y;
	return 0;
}

int ustc_SubImgMatch_hist(Mat grayImg,Mat subImg,int *x,int *y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "the image is NULL" << endl;
		return 0;
	}
	if (grayImg.channels() != 1 || subImg.channels() != 1)
	{
		cout << " the format is incorrect" << endl;
		return 0;
	}
		int referencehist[256];
		int bigImghist[256];
		ustc_CalcHist(subImg, referencehist, 256);
		int height = grayImg.rows;
		int width = grayImg.cols;
		int sub_height = subImg.rows;
		int sub_width = subImg.cols;
		int graysize = height*width;
		int subsize = sub_height*sub_width;
		if (graysize < subsize || height < sub_height || width < sub_width)
		{
			cout << "the subImg can not match the measurement of the bigImg" << endl;
			return 0;
		}
		int match_difference = INT16_MAX;
		int match_x = 0;int match_y = 0;
		for (int i = 0;i < height - sub_height;i++)
		{
			for (int j = 0;j < width - sub_width;j++)
			{
				for (int l = 0;l < 256;l++)
					bigImghist[l] = 0;
				for (int m = 0;m < sub_height;m++)
				{
					for (int n = 0;n < sub_width;n++)
					{
						int row_index = i + m;
						int col_index = j + n;
						int grayImgpix = grayImg.data[row_index*width + col_index];
						bigImghist[grayImgpix]++;
					}
				}
				int total_difference = 0;
				for (int k = 0;k < 256;k++)
				{
					int a = bigImghist[k];
					int b = referencehist[k];
					if (a > b)
						total_difference += a - b;
					else total_difference += b - a;
				}
				if (total_difference < match_difference)
				{
					match_difference = total_difference;
					match_x = i;
					match_y = j;
				}
			}
		}
        
		printf("the vexpoint position is (%d,%d)", match_x, match_y);
		x = &match_x;
		y = &match_y;
		return 0;
	}
