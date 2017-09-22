
#include "SubImageMatch.h"


int ustc_ConvertBgr2Gray(Mat bgrImg,Mat&grayImg)
{
	if (NULL == bgrImg.data)
	{
		cout << "colorImg ERROR! ---ustc_ConvertBgr2Gray" << endl;
		return -1;
	}
	if (grayImg.channels() != 3)
	{
		cout << "colorImg channels ERROR!   ----ustc_ConvertBgr2Gray" << endl;
		return -3;
	}
	int my_row, my_col,graystep,colorstep;
	int Height = bgrImg.rows;
	int Width = bgrImg.cols;

	
	for (my_row = 0; my_row < Height; my_row++)
	{
		int temp = my_row * Width;
		for (my_col = 0; my_col < Width; my_col++)
		{
			graystep = temp + my_col;
			colorstep = 3 * graystep;
			int Blue = bgrImg.data[colorstep+0];
			int Green = bgrImg.data[colorstep + 1];
			int Red = bgrImg.data[colorstep+2];
			int Gray = (Red * 306 + Green * 601 + Blue * 117)>>10;
			grayImg.data[graystep] = Gray;
		}
	}
	
	namedWindow("ConvertBgr2Gray");
	imshow("ConvertBgr2Gray",grayImg);
	waitKey(1);


	return 0;
}   

int ustc_CalcGrad(Mat grayImg,Mat&gradImg_x,Mat&gradImg_y)
{
	if (NULL == grayImg.data)
	{
		cout << "GrayImg ERROR! ---ustc_CalcGrad" << endl;
		return -1;
	}
	if (grayImg.channels() != 1)
	{
		cout << "grayImg channels ERROR!   ----ustc_CalcGrad" << endl;
		return -3;
	}
	
	int Height = grayImg.rows;
	int Width = grayImg.cols;
	int my_row, my_col;

	gradImg_x.setTo(0);
	gradImg_y.setTo(0);

	for (my_row = 1; my_row < Height - 1; my_row++)
	{
		int temp = my_row * Width;
		for (my_col = 1; my_col < Width - 1; my_col++)
		{
			int step = temp + my_col;
			int stepup = step - Width;
			int stepdown = step + Width;
			int stepleft = step - 1;
			int stepright = step + 1;
			
			int grad_x = 2 * grayImg.data[stepright] + grayImg.data[stepright - Width] + grayImg.data[stepright + Width] - 2*grayImg.data[stepleft] - grayImg.data[stepleft + Width] - grayImg.data[stepleft - Width];
			int grad_y = 2 * grayImg.data[stepdown] + grayImg.data[stepdown + 1] + grayImg.data[stepdown - 1] - 2 * grayImg.data[stepup] - grayImg.data[stepup - 1] - grayImg.data[stepup + 1];
			grad_x = (grad_x >= 0 ? grad_x : -grad_x);
			grad_y = (grad_y >= 0 ? grad_y : -grad_y);

			gradImg_x.data[step] = grad_x;
			gradImg_y.data[step] = grad_y;
		}
	}
	
	return 0;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat&angleImg, Mat&magImg)
{
	if (NULL == gradImg_x.data)
	{
		cout << "ERROR!!  ---func3" << endl;
	}
	if (NULL == gradImg_y.data)
	{
		cout << "ERROR!!  ---func3" << endl;
	}
	double tan_gram[900];
	float temp_angle = 0;
	for (register int i = 0; i < 900; i++)
		{
			tan_gram[i] = tan((double)(i) / 10.0);
		}

	int Height = gradImg_x.rows;
	int Width = gradImg_x.cols;
	int count = Height*Width;

	for (register int place = 0; place < count; place++)
	{
		int grad_x = gradImg_x.data[place];
		int grad_y = gradImg_y.data[place];
		
		float temp_tan = (float)(grad_y) / (float)(grad_x);
		temp_tan = (temp_tan >= 0 ? temp_tan : -temp_tan);
		int flag = 0;
		for (int i = 0; i < 900 && flag == 0; i++)
		{
			if (temp_tan < tan_gram[i])
			{
				temp_angle = i / 10.0;
				flag = 1;
			}
		}
		if (grad_x >= 0 && grad_y >= 0)
		{
			angleImg.data[place] = temp_angle;
		}
		else if (grad_x < 0 && grad_y >= 0)
		{
			angleImg.data[place] = 180.0 - temp_angle;
		}
		else if (grad_x < 0 && grad_y < 0)
		{
			angleImg.data[place] = 180.0 + temp_angle;
		}
		else if (grad_x >= 0 && grad_y < 0)
		{
			angleImg.data[place] = 360.0 - temp_angle;
		}

		float temp_mag2 = grad_x*grad_x + grad_y*grad_y;
		int t = *(int*)&temp_mag2;
		t -= 1 << 23;
		t >>= 1;
		t += 1 << 29;
		magImg.data[place] = *(float*)&t;

	}
	return 0;
}

int ustc_Threshold(Mat grayImg,Mat&binaryImg,int th)
{
	if (NULL == grayImg.data)
	{
		cout << "GrayImg ERROR! ---ustc_Threshold" << endl;
		return -1;
	}
	if (grayImg.channels() != 1)
	{
		cout << "grayImg channels ERROR!   ----ustc_Threshold" << endl;
		return -3;
	}
	binaryImg.setTo(0);
	int val=0;
	int Height = grayImg.rows;
	int Width = grayImg.cols;
	
	for (int my_row=0; my_row < Height; my_row++)
	{
		int temp = Width * my_row;
		for (int my_col=0; my_col < Width; my_col++)
		{
			int step = temp + my_col;
			int gray = grayImg.data[step];
			if (gray-th>>31 )    //gray>=th
			{
				val = 255;
			}
			else //(gray < th)
			{
				val = 0;
			}
			binaryImg.data[step] = val;
		}
	}
	
	namedWindow("Threshold");
	imshow("Threshold", binaryImg);
	waitKey(1);
	return 0;
}

int ustc_CalcHist(Mat grayImg, int*hist, int hist_len)
{
	if (NULL == grayImg.data)
	{
		cout << "GrayImg ERROR! ---ustc_CalcHist" << endl;
		return -1;
	}
	if (grayImg.channels() != 1)
	{
		cout << "grayImg channels ERROR!   ----ustc_CalcHist" << endl;
		return -3;
	}

	for (int i = 0; i < hist_len; i++)
	{
		hist[i] = 0;
	}
	int Height = grayImg.rows;
	int Width = grayImg.cols;
	
	for (int my_row = 0; my_row < Height; my_row++)
	{
		int temp = my_row * Width;
		for (int my_col = 0; my_col < Width; my_col++)
		{
			int place = temp + my_col;
			hist[grayImg.data[place]]++;
		}
	}
	return 0;
}

int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int*x, int*y)
{
	if (NULL == grayImg.data)
	{
		cout << "grayImg ERROR!!  ---ustc_SubImgMatch_gray" << endl;
		return -1;
	}
	if (NULL == subImg.data)
	{
		cout << "subImg ERROR!!  ---ustc_SubImgMatch_gray" << endl;
		return -2;
	}
	int gray_channels = grayImg.channels();
	int sub_channels = subImg.channels();
	if (gray_channels != sub_channels)
	{
		cout << "Different Channels.Can not match  ---ustc_SubImgMatch_gray" << endl;
		return -3;
	}
	int Height = grayImg.rows;
	int Width = grayImg.cols;
	int length_x = subImg.cols;
	int length_y = subImg.rows;

	int max_x = Width - length_x;
	int max_y = Height - length_y;

	int Gray = 256 * length_x*length_y, temp_Gray=0;
	
	for (int my_row = 0; my_row < max_y; my_row++)
	{
		int temp = my_row * Width;

		for (int my_col = 0; my_col < max_x; my_col++)
		{
			int place = temp + my_col;
			for (int sub_row = 0,temp_Gray=0; sub_row < length_y; sub_row++)
			{
				int temp_sub = sub_row * length_x;
				for (int sub_col = 0; sub_col < length_x; sub_col++)
				{
					int sub_place = temp_sub + sub_col;
					temp_Gray += abs(grayImg.data[place + sub_row * Width + sub_col]-subImg.data[sub_place]);
				}
			}
			if (Gray >= temp_Gray)
			{
				Gray = temp_Gray;
				*x = my_col;
				*y = my_row;
			}
		}
	}
	return 0;
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int*x, int*y)
{
	if (NULL == colorImg.data)
	{
		cout << "colorImg ERROR!!  ---ustc_SubImgMatch_bgr" << endl;
		return -1;
	}
	if (NULL == subImg.data)
	{
		cout << "subImg ERROR!!  ---ustc_SubImgMatch_bgr" << endl;
		return -2;
	}
	int gray_channels = colorImg.channels();
	int sub_channels = subImg.channels();
	if (gray_channels != sub_channels)
	{
		cout << "Different Channels.Can not match  ---ustc_SubImgMatch_bgr" << endl;
		return -3;
	}
	int Height = colorImg.rows;
	int Width = colorImg.cols;
	int length_x = subImg.cols;
	int length_y = subImg.rows;
	int Color = 3 * 256 * length_x*length_y , temp_Color = 0;

	int max_x = Width - length_x;
	int max_y = Height - length_y;

	for (int my_row = 0; my_row < max_y; my_row++)
	{
		int temp = my_row * Width;
		for (int my_col = 0; my_col < max_x; my_col++)
		{
			int place = temp + my_col;

			for (int sub_row = 0,temp_Color=0; sub_row < length_y; sub_row++)
			{
				int sub_temp = sub_row * length_x;
				for (int sub_col = 0; sub_col < length_y; sub_col++)
				{
					int sub_place = 3*(sub_temp + sub_col);
					int color_place = 3 * (place + sub_row*Width + sub_col);
					temp_Color += abs(colorImg.data[color_place] - subImg.data[sub_place]) + abs(colorImg.data[color_place + 1]- subImg.data[sub_place + 1]) + abs(colorImg.data[color_place + 2]   - subImg.data[sub_place + 2]);
				}
			}
			if (Color >= temp_Color)
			{
				Color = temp_Color;
				*x = my_col;
				*y = my_row;
			}
		}
	}
	return 0;
}

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int*x, int*y) 
{
	
	
	if (NULL == grayImg.data)
	{
		cout << "grayImg ERROR!!  ---ustc_SubImgMatch_corr" << endl;
		return -1;
	}
	if (NULL == subImg.data)
	{
		cout << "subImg ERROR!!  ---ustc_SubImgMatch_corr" << endl;
		return -2;
	}
	int gray_channels = grayImg.channels();
	int sub_channels = subImg.channels();
	if (gray_channels != sub_channels)
	{
		cout << "Different Channels.Can not match  ---ustc_SubImgMatch_corr" << endl;
		return -3;
	}


	int Height = grayImg.rows;
	int Width = grayImg.cols;
	int length_x = subImg.cols;
	int length_y = subImg.rows;
	int sub_count = length_x*length_y;

	int max_x = Width - length_x;
	int max_y = Height - length_y;

	float R = 10000000, temp_R=0;

	float sum = 0, sum1 = 0, sum2 = 0;
	for (int i = 0; i < sub_count; i++)
	{
		sum += subImg.data[i];
	}
	
	int t = *(int*)&sum;
	t -= 1 << 23;
	t >>= 1;
	t += 1 << 29;
	float temp_2 = *(float*)&t;



	for (int my_row = 0; my_row < max_y; my_row++)
	{
		int temp = my_row * Width;

		for (int my_col = 0; my_col < max_x; my_col++)
		{
			int place = temp + my_col;
			for (int sub_row = 0, sum1=0,sum2=0,temp_R=0; sub_row < length_y; sub_row++)
			{
				int temp_sub = sub_row * length_x;
				int temp_gray = sub_row*Width;
				for (int sub_col = 0; sub_col < length_x; sub_col++)
				{
					int sub_place = temp_sub + sub_col;
					sum1 += grayImg.data[place + temp_gray + sub_col];
					sum2 += subImg.data[sub_place] * grayImg.data[place + temp_gray + sub_col];
										
				}
			}
			
			int s = *(int*)&sum1;
			s -= 1 << 23;
			s >>= 1;
			s += 1 << 29;
			float temp_3 = *(float*)&s;

			int r = *(int*)&sum2;
			r -= 1 << 23;
			r >>= 1;
			r += 1 << 29;
			float temp_1= *(float*)&r;

			temp_R = temp_1 / temp_2 / temp_3;

			if (temp_R <= 1)
			{
				temp_R = 1.0 / temp_R;
			}

			if (temp_R<R)
			{
				R=temp_R;
				*x = my_col;
				*y = my_row;
			}
		}
	}


	return 0;
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int*x, int*y)
{

	if (NULL == grayImg.data)
	{
		cout << "grayImg ERROR!!  ---ustc_SubImgMatch_angle" << endl;
		return -1;
	}
	if (NULL == subImg.data)
	{
		cout << "subImg ERROR!!  ---ustc_SubImgMatch_angle" << endl;
		return -2;
	}
	int gray_channels = grayImg.channels();
	int sub_channels = subImg.channels();
	if (gray_channels != sub_channels)
	{
		cout << "Different Channels.Can not match  ---ustc_SubImgMatch_angle" << endl;
		return -3;
	}

	double tan_gram[900];
	float temp_angle = 0;
	for (register int i = 0; i < 900; i++)
	{
		tan_gram[i] = tan((double)(i) / 10.0);
	}

	int Height = grayImg.rows;
	int Width = grayImg.cols;
	int length_x = subImg.cols;
	int length_y = subImg.rows;
	int max_y = Height - length_y;
	int max_x = Width - length_x;

	float Angle = 360 * length_x*length_y;
	int result = 0;
	
	Mat grayImg_grad_x(Height, Width, CV_32F);
	Mat grayImg_grad_y(Height, Width, CV_32F);
	Mat subImg_grad_x(length_y, length_x, CV_32F);
	Mat subImg_grad_y(length_y, length_x, CV_32F);
	Mat grayImg_Angle(Height, Width, CV_32F);
	Mat subImg_Angle(length_y, length_x, CV_32F);


	ustc_CalcGrad(grayImg, grayImg_grad_x, grayImg_grad_y);
	ustc_CalcGrad(subImg, subImg_grad_x,subImg_grad_y);

	
	int count1 = Height*Width;
	float temp_angle1;
	for (register int place1 = 0; place1 < count1; place1++)
	{
		int grad_x = grayImg_grad_x.data[place1];
		int grad_y = grayImg_grad_y.data[place1];

		float temp_tan = (float)(grad_y) / (float)(grad_x);
		temp_tan = (temp_tan >= 0 ? temp_tan : -temp_tan);
		int flag = 0;
		for (int i = 0; i < 900 && flag == 0; i++)
		{
			if (temp_tan < tan_gram[i])
			{
				temp_angle1 = i / 10.0;
				flag = 1;
			}
		}
		if (grad_x >= 0 && grad_y >= 0)
		{
			grayImg_Angle.data[place1] = temp_angle1;
		}
		else if (grad_x < 0 && grad_y >= 0)
		{
			grayImg_Angle.data[place1] = 180.0 - temp_angle1;
		}
		else if (grad_x < 0 && grad_y < 0)
		{
			grayImg_Angle.data[place1] = 180.0 + temp_angle1;
		}
		else if (grad_x >= 0 && grad_y < 0)
		{
			grayImg_Angle.data[place1] = 360.0 - temp_angle1;
		}

	}

	float temp_angle2;
	int count2 = length_x*length_y;
	for (register int place2 = 0; place2 < count2; place2++)
	{
		int grad_x = subImg_grad_x.data[place2];
		int grad_y = subImg_grad_y.data[place2];

		float temp_tan2 = (float)(grad_y) / (float)(grad_x);
		temp_tan2 = (temp_tan2 >= 0 ? temp_tan2 : -temp_tan2);
		int flag = 0;
		for (int i = 0; i < 900 && flag == 0; i++)
		{
			if (temp_tan2 < tan_gram[i])
			{
				temp_angle2 = i / 10.0;
				flag = 1;
			}
		}
		if (grad_x >= 0 && grad_y >= 0)
		{
			subImg_Angle.data[place2] = temp_angle2;
		}
		else if (grad_x < 0 && grad_y >= 0)
		{
			subImg_Angle.data[place2] = 180.0 - temp_angle2;
		}
		else if (grad_x < 0 && grad_y < 0)
		{
			subImg_Angle.data[place2] = 180.0 + temp_angle2;
		}
		else if (grad_x >= 0 && grad_y < 0)
		{
			subImg_Angle.data[place2] = 360.0 - temp_angle2;
		}

	}

	int place = 0;

	for (int my_row=0;my_row<max_y;my_row++)
	{
		int temp = my_row*Width;
		for (int my_col = 0; my_col < max_y; my_col++)
		{
			place = temp + my_col;
			
			for (int sub_row = 0,temp_angle=0.0; sub_row < length_y; sub_row++)
			{
				int temp_sub = sub_row*length_x;
				int temp_sub2 = sub_row*Width;
				for (int sub_col = 0; sub_col < length_x; sub_col++)
				{
					int sub_place = temp_sub + sub_col;
					float temp_now_angle=abs(subImg_Angle.data[sub_place] - grayImg_Angle.data[place + temp_sub2 + sub_col]);
					if (temp_now_angle > 180)
					{
						temp_now_angle = 360 - temp_now_angle;
					}

					temp_angle += temp_now_angle;
				}
			}

			if (temp_angle < Angle)
			{
				Angle = temp_angle;
				*x = my_col;
				*y = my_row;
			}

		}
	}

	return 0;
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int*x, int*y)
{
	if (NULL == grayImg.data)
	{
		cout << "grayImg ERROR!!  ---ustc_SubImgMatch_mag" << endl;
		return -1;
	}
	if (NULL == subImg.data)
	{
		cout << "subImg ERROR!!  ---ustc_SubImgMatch_mag" << endl;
		return -2;
	}
	int gray_channels = grayImg.channels();
	int sub_channels = subImg.channels();
	if (gray_channels != sub_channels)
	{
		cout << "Different Channels.Can not match  ---ustc_SubImgMatch_mag" << endl;
		return -3;
	}

	int Height = grayImg.rows;
	int Width = grayImg.cols;
	int length_x = subImg.cols;
	int length_y = subImg.rows;
	int max_y = Height - length_y;
	int max_x = Width - length_x;

	int Mag = 1020 * length_x*length_y, temp_mag = 0;
	int result = 0;

	Mat grayImg_grad_x(Height, Width, CV_32F);
	Mat grayImg_grad_y(Height, Width, CV_32F);
	Mat subImg_grad_x(length_y, length_x, CV_32F);
	Mat subImg_grad_y(length_y, length_x, CV_32F);
	Mat grayImg_Mag(Height, Width, CV_32F);
	Mat subImg_Mag(length_y, length_x, CV_32F);

	ustc_CalcGrad(grayImg, grayImg_grad_x, grayImg_grad_y);
	ustc_CalcGrad(subImg, subImg_grad_x, subImg_grad_y);

	int count1 = Height*Width;
	for (register int place1 = 0; place1 < count1; place1++)
	{
		int grad_x = grayImg_grad_x.data[place1];
		int grad_y = grayImg_grad_y.data[place1];

		float temp_mag1 = grad_x*grad_x + grad_y*grad_y;
		int t = *(int*)&temp_mag1;
		t -= 1 << 23;
		t >>= 1;
		t += 1 << 29;
		subImg_Mag.data[place1] = *(float*)&t;
	}

	int count2 = length_x*length_y;
	for (register int place2 = 0; place2 < count2; place2++)
	{
		int grad_x = grayImg_grad_x.data[place2];
		int grad_y = grayImg_grad_y.data[place2];

		float temp_mag2 = grad_x*grad_x + grad_y*grad_y;
		int t = *(int*)&temp_mag2;
		t -= 1 << 23;
		t >>= 1;
		t += 1 << 29;
		subImg_Mag.data[place2] = *(float*)&t;
	}


	int place = 0;

	for (int my_row = 0; my_row<max_y; my_row++)
	{
		int temp = my_row*Width;
		for (int my_col = 0; my_col < max_y; my_col++)
		{
			place = temp + my_col;

			for (int sub_row = 0, temp_mag = 0; sub_row < length_y; sub_row++)
			{
				int temp_sub = sub_row*length_x;
				int temp_sub2 = sub_row*Width;
				for (int sub_col = 0; sub_col < length_x; sub_col++)
				{
					int sub_place = temp_sub + sub_col;
					temp_mag += abs(subImg_Mag.data[sub_place] - grayImg_Mag.data[place + temp_sub2 + sub_col]);
				}
			}

			if (temp_mag < Mag)
			{
				Mag = temp_mag;
				*x = my_col;
				*y = my_row;
			}

		}
	}

	return 0;
}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int*x, int*y)
{
	if (NULL == grayImg.data)
	{
		cout << "grayImg ERROR!!  ---ustc_SubImgMatch_hist" << endl;
		return -1;
	}
	if (NULL == subImg.data)
	{
		cout << "subImg ERROR!!  ---ustc_SubImgMatch_hist" << endl;
		return -2;
	}
	int gray_channels = grayImg.channels();
	int sub_channels = subImg.channels();
	if (gray_channels != sub_channels)
	{
		cout << "Different Channels.Can not match  ---ustc_SubImgMatch_hist" << endl;
		return -3;
	}

	int Height = grayImg.rows;
	int Width = grayImg.cols;
	int sub_height = subImg.rows;
	int sub_width = subImg.cols;
	int max_row = Height - sub_height;
	int max_col = Width - sub_width;
	
	int temp_gray_hist[256];
	int sub_hist[256];

	Mat temp_grayImg(sub_height, sub_width, CV_8UC1);

	ustc_CalcHist(subImg,sub_hist,256);

	int Hist = 256 * sub_height*sub_width, temp_hist=0;

	for (int my_row = 0; my_row < max_row; my_row++)
	{
		int temp = my_row * Width;
		for (int my_col = 0; my_col < max_col; my_col++)
		{
			int place = temp + my_col;
			

			for (int sub_row = 0; sub_row < sub_height; sub_row++)
			{
				int sub_temp = sub_row*sub_width;
				int sub_temp2 = sub_row*sub_width;
				for (int sub_col = 0; sub_col < sub_width; sub_col++)
				{
					int sub_place = sub_temp + sub_col;
					int sub_temp2 = sub_row*sub_width;
					temp_grayImg.data[sub_place] = grayImg.data[place + sub_temp2 + sub_col];
				}
			}
			ustc_CalcHist(temp_grayImg,temp_gray_hist, 256);

			for (int i = 0,temp_hist=0; i < 256; i++)
			{
				temp_hist += abs(temp_gray_hist[i] - sub_hist[i]);
			}

			if (Hist > temp_hist)
			{
				Hist = temp_hist;
				*x = my_col;
				*y = my_row;
			}
		}
	}
	return 0;

}
