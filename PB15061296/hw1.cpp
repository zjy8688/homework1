#include<SubImageMatch.h>

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg) {   
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	if (bgrImg.channels() != 3|| grayImg.channels() != 1) {
		cout << "channels is warniing." << endl;
		return -1;
	}
	int width = bgrImg.cols;
	int height = bgrImg.rows;
	int width_gray = grayImg.cols;
	int height_gray = grayImg.rows;
	if (width > width_gray || height > height_gray) {
		cout << "gray wrong." << endl;
		return -1;
	}
	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp = row_i * width;
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int temp1 = 3 * (temp + col_j);
			int b = bgrImg.data[temp1 ];
			int g = bgrImg.data[temp1 + 1];
			int r = bgrImg.data[temp1 + 2];

			int grayVal = b * 116 + g * 601 + r * 307;
			grayImg.data[temp + col_j] = (grayVal >> 10);
		}
	}
	return 1;
}


int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y) {        
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	if (gradImg_x.channels() != 1 || gradImg_y.channels() != 1 ||grayImg.channels() != 1) {
		cout << "channels is warniing." << endl;
		return -1;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int width_x = gradImg_x.cols;
	int height_x = gradImg_x.rows;
	int width_y = gradImg_y.cols;
	int height_y = gradImg_y.rows;
	if (width > width_x || width > width_y || height > height_x || height > height_y) {
		cout << "grad_x OR grad_y wrong." << endl;
		return -1;
	}

	gradImg_x.setTo(0);
	gradImg_y.setTo(0);

	
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		int temp0 = (row_i - 1) * width;
		int temp1 = row_i* width ;
		int temp2 = (row_i + 1)* width;
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int g1 = grayImg.data[temp0 + col_j - 1];
			int g2 = grayImg.data[temp0 + col_j + 1];
			int g3 = grayImg.data[temp2 + col_j - 1];
			int g4 = grayImg.data[temp2 + col_j + 1];
			int grad_x =
				g2+ 2 * grayImg.data[temp1 + col_j + 1]+ g4 
				- g1- 2 * grayImg.data[temp1 + col_j - 1]
				- g3; 

			int grad_y =
				g3+ 2 * grayImg.data[temp2 + col_j] 
				+ g4- g1- 2 * grayImg.data[temp0 + col_j]
				- g2;

			((float*)gradImg_x.data)[temp1 + col_j] = grad_x;
			((float*)gradImg_y.data)[temp1 + col_j] = grad_y;
		}
	}
	return 1;
}



int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg) {      
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	if (gradImg_x.channels() != 1 || gradImg_y.channels() != 1|| angleImg.channels() != 1|| magImg.channels() != 1) {
		cout << "channels is warniing." << endl;
		return -1;
	}

	int width = gradImg_x.cols;
	int height = gradImg_x.rows;
	int width_y = gradImg_y.cols;
	int height_y = gradImg_y.rows;
	int width_angle = angleImg.cols;
	int height_angle = angleImg.rows;
	int width_mag = magImg.cols;
	int height_mag = magImg.rows;

	if (width != width_y || height != height_y) {
		cout << "grad_x OR grad_y wrong." << endl;
		return -1;
	}
	if (width > width_angle || width > width_mag || height > height_angle || height > height_mag) {
		cout << "angle OR mag wrong." << endl;
		return -1;
	}

	angleImg.setTo(0);
	Mat F_gradImg_x = gradImg_x;
	Mat F_gradImg_y = gradImg_y;
	Mat F_angleImg = angleImg;
	Mat F_magImg = magImg;


	
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		int temp = row_i * width;
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			int temp1 = temp + col_j;
			float grad_x = ((float*)F_gradImg_x.data)[temp1];
			float grad_y = ((float*)F_gradImg_y.data)[temp1];
			float grad_xa, grad_ya;
			if (grad_x < 0) grad_xa = -grad_x;
			else grad_xa = grad_x;
			if (grad_y < 0) grad_ya = -grad_y;
			else grad_ya = grad_y;
			float angle = (grad_ya * 90) / (grad_xa + grad_ya);
			if (angle > 45) angle += 0.0079*(90 - angle)*(angle - 45);
			else angle -= 0.0079*angle*(45 - angle);
			if (grad_x < 0) angle = 180 - angle;
			if (grad_y < 0) angle = 360 - angle;
			float x = grad_y*grad_y + grad_x*grad_x;
			int t = *(int*)&x;
			t -= 0x3f800000;
			t >>= 1;
			t += 0x3f800000;
			x = *(float*)&t;
			float mag = x;
			((float*)F_angleImg.data)[temp1] = angle;
			((float*)F_magImg.data)[temp1] = mag;
		}
	}
	angleImg = F_angleImg;
	magImg = F_magImg;
	return 1;
}

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th) {       
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	if (grayImg.channels() != 1|| binaryImg.channels()!=1) {
		cout << "channels is warniing." << endl;
		return -1;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int binary_width = binaryImg.cols;
	int binary_height = binaryImg.rows;
	if (width > binary_width || height > binary_height) {
		cout << "binaryImg too samll." << endl;
		return -1;
	}
	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp0 = row_i * width;
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int temp1 = temp0 + col_j;
			binaryImg.data[temp1]  = (((th - grayImg.data[temp1]) >> 31)&1) * 255;
		}
	}
	return 1;

}


int ustc_CalcHist(Mat grayImg, int* hist, int hist_len) {           
	if (NULL == grayImg.data || NULL == hist)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	if (grayImg.channels() != 1 ) {
		cout << "channels is warniing." << endl;
		return -1;
	}
	if (hist_len < 256) {
		cout << "hist_len is warniing." << endl;
		return -1;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;

	
	for (int i = 0; i < hist_len; i++)
	{
		hist[i] = 0;
	}

	
	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp = row_i * width;
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int pixVal = grayImg.data[temp + col_j];
			hist[pixVal]++;
		}
	}
	return 1;
}


int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int Min=0x11111111;

	if (grayImg.channels() != 1 || subImg.channels() != 1) {
		cout << "channels is warniing." << endl;
		return -1;
	}
	if (sub_width > width || sub_height > height) {
		cout << "subImg too huge" << endl;
		return -1;
	}

	Mat searchImg = grayImg;
	Mat searchsub = subImg;

	int a, b;
	
	for (int i = 0; i < height - sub_height; i++)
	{
		int temp = i*width;
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			
			int temp1 = temp + j;
			for (int in_x = 0; in_x < sub_height; in_x++)
			{
				int temp2 = temp1 + in_x*width;
				int sub = in_x*sub_height;
				for (int in_y = 0; in_y < sub_width; in_y++)
				{
					
					int pix = searchImg.data[temp2 + in_y] - searchsub.data[sub + in_y];
					total_diff -= (2 * (pix >> 31 & 1) - 1)*pix;
				}
			}
			
			if (Min > total_diff) {
				Min = total_diff;
				a = j;
				b = i;
			}
		}
	}
	*x = a;
	*y = b;
	return 1;
}


int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int *x, int *y) {
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}

	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int Min = 0x00ffffff;

	if (colorImg.channels() != 3 || subImg.channels() != 3) {
		cout << "channels is warniing." << endl;
		return -1;
	}
	if (sub_width > width || sub_height > height) {
		cout << "subImg too huge" << endl;
		return -1;
	}

	
	Mat searchImg=colorImg;
	Mat searchsub = subImg;
	
	int a, b;
	
	for (int i = 0; i < height - sub_height; i++)
	{
		int temp = 3*i*width;
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			
			int temp1 = temp + 3*j;
			for (int in_x = 0; in_x < sub_height; in_x++)
			{
				int temp2 = temp1 + 3*in_x*width;
				int sub = 3*in_x*sub_height;
				for (int in_y = 0; in_y < sub_width; in_y++)
				{
					
					int temp3 = temp2 + 3 * in_y;
					int sub1 = sub + 3 * in_y;
					int pix0 = searchImg.data[temp3] - searchsub.data[sub1];
					int pix1 = searchImg.data[temp3+1] - searchsub.data[sub1+1];
					int pix2 = searchImg.data[temp3+2] - searchsub.data[sub1+2];
					total_diff -= (2 * (pix0 >> 31 & 1) - 1)*pix0+ (2 * (pix1 >> 31 & 1) - 1)*pix1+ (2 * (pix2 >> 31 & 1) - 1)*pix2;
				}
			}
			
			if (Min > total_diff) {
				Min = total_diff;
				a = j;
				b = i;
			}
		}
	}
	*x = a;
	*y = b;
	return 1;
}


int ustc_SubImgMatch_corr(Mat grayImg,Mat subImg,int* x,int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	float Max = 0;

	if (grayImg.channels() != 1 || subImg.channels() != 1) {
		cout << "channels is warniing." << endl;
		return -1;
	}
	if (sub_width > width || sub_height > height) {
		cout << "subImg too huge" << endl;
		return -1;
	}

	Mat searchImg = grayImg;
	Mat searchsub = subImg;

	int a, b;
	for (int i = 0; i < height - sub_height; i++)
	{
		int temp = i*width;
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_G = 0;
			int total_S = 0;
			int total_SG = 0;
			float total_diff = 0;
			int temp1 = temp + j;
			for (int in_x = 0; in_x < sub_height; in_x++)
			{
				int temp2 = temp1 + in_x*width;
				int sub = in_x*sub_height;
				for (int in_y = 0; in_y < sub_width; in_y++)
				{
					int pix_gary = searchImg.data[temp2 + in_y];
					int pix_sub = searchsub.data[sub + in_y];
					total_G += pix_gary*pix_gary;
					total_S += pix_sub*pix_sub;
					total_SG += pix_sub*pix_gary;
				}
			}
			total_diff = float(total_SG) / (sqrt(total_S)*sqrt(total_G));
			if (Max < total_diff) {
				Max = total_diff;
				a = j;
				b = i;
			}
		}
	}
	*x = a;
	*y = b;
	return 1;
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	float Min = 0xffffffff;

	if (grayImg.channels() != 1 || subImg.channels() != 1) {
		cout << "channels is warniing." << endl;
		return -1;
	}
	if (sub_width > width || sub_height > height) {
		cout << "subImg too huge" << endl;
		return -1;
	}

	Mat searchImg = grayImg;
	Mat searchsub = subImg;
	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);
	Mat subImg_x(sub_height, sub_width, CV_32FC1);
	Mat subImg_y(sub_height, sub_width, CV_32FC1);
	Mat sub_angle(sub_height, sub_width, CV_8UC1);
	Mat gray_angle(height, width, CV_8UC1);
	int sub_x, sub_y;
	sub_x = ustc_CalcGrad(searchImg, gradImg_x, gradImg_y);
	sub_y = ustc_CalcGrad(searchsub, subImg_x, subImg_y);
	int row_i, col_j, temp_sub, temp1_sub;
	float grad_x_sub, grad_y_sub, grad_xa, grad_ya, angle_sub;
	for (row_i = 1; row_i < sub_height - 1; row_i++)     
	{
		temp_sub = row_i * sub_width;
		for (col_j = 1; col_j < sub_width - 1; col_j += 1)
		{
			temp1_sub = temp_sub + col_j;
			grad_x_sub = ((float*)subImg_x.data)[temp1_sub];
			grad_y_sub = ((float*)subImg_y.data)[temp1_sub];
			if (grad_x_sub < 0) grad_xa = -grad_x_sub;
			else grad_xa = grad_x_sub;
			if (grad_y_sub < 0) grad_ya = -grad_y_sub;
			else grad_ya = grad_y_sub;
			angle_sub = (grad_ya * 90) / (grad_xa + grad_ya);
			if (angle_sub > 45) angle_sub += 0.0079*(90 - angle_sub)*(angle_sub - 45);
			else angle_sub -= 0.0079*angle_sub*(45 - angle_sub);
			if (grad_x_sub < 0) angle_sub = 180 - angle_sub;
			if (grad_y_sub < 0) angle_sub = 360 - angle_sub;
			(sub_angle.data)[temp1_sub] = *(int*)&(angle_sub)/2;
		}
	}
	for (row_i = 1; row_i < height - 1; row_i++)                    
	{
		temp_sub = row_i * width;
		for (col_j = 1; col_j < width - 1; col_j += 1)
		{
			temp1_sub = temp_sub + col_j;
			grad_x_sub = ((float*)gradImg_x.data)[temp1_sub];
			grad_y_sub = ((float*)gradImg_y.data)[temp1_sub];
			if (grad_x_sub < 0) grad_xa = -grad_x_sub;
			else grad_xa = grad_x_sub;
			if (grad_y_sub < 0) grad_ya = -grad_y_sub;
			else grad_ya = grad_y_sub;
			angle_sub = (grad_ya * 90) / (grad_xa + grad_ya);
			if (angle_sub > 45) angle_sub += 0.0079*(90 - angle_sub)*(angle_sub - 45);
			else angle_sub -= 0.0079*angle_sub*(45 - angle_sub);
			if (grad_x_sub < 0) angle_sub = 180 - angle_sub;
			if (grad_y_sub < 0) angle_sub = 360 - angle_sub;
			(gray_angle.data)[temp1_sub] = *(int*)&(angle_sub) / 2;
		}
	}
	int a, b;
	for(int i = 0; i < height - sub_height; i++)
	{
		int temp = i*width;
		for (int j = 0; j < width - sub_width; j++)
		{
			float total_diff = 0;
			int temp1 = temp + j;
			for (int in_x = 1; in_x < sub_height-1; in_x++)
			{
				int temp2 = temp1 + in_x*width;
				int sub = in_x*sub_height;
				for (int in_y = 1; in_y < sub_width-1; in_y++)
				{
					int temp3 = temp2 + in_y;
					int pix= sub_angle.data[sub + in_y] - gray_angle.data[temp3];
					pix= (1-2 * ((pix >> 31) & 1))*pix;
					pix = 90 + (2 * (((pix - 90) >> 31) & 1) - 1)*(pix-90);    
					total_diff += pix;
				}
			}
			if (Min > total_diff) {
				Min = total_diff;
				a = j;
				b = i;
			}
		}
	}
	*x = a;
	*y = b;
	return 1;
}



int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	float Min = 0xffffffff;

	if (grayImg.channels() != 1 || subImg.channels() != 1) {
		cout << "channels is warniing." << endl;
		return -1;
	}
	if (sub_width > width || sub_height > height) {
		cout << "subImg too huge" << endl;
		return -1;
	}

	Mat searchImg = grayImg;
	Mat searchsub = subImg;
	Mat gradImg_x(height, width, CV_32FC1);
	Mat gradImg_y(height, width, CV_32FC1);
	Mat subImg_x(sub_height, sub_width, CV_32FC1);
	Mat subImg_y(sub_height, sub_width, CV_32FC1);
	Mat sub_mag(sub_height, sub_width, CV_8UC1);
	Mat gray_mag(height, width, CV_8UC1);
	int sub_x, sub_y;
	sub_x = ustc_CalcGrad(searchImg, gradImg_x, gradImg_y);
	sub_y = ustc_CalcGrad(searchsub, subImg_x, subImg_y);
	
	int row_i, col_j, temp_sub, temp1_sub;
	float grad_x_sub, grad_y_sub,mag_sub;
	for (row_i = 1; row_i < sub_height - 1; row_i++)     
	{
		temp_sub = row_i * sub_width;
		for (col_j = 1; col_j < sub_width - 1; col_j += 1)
		{
			temp1_sub = temp_sub + col_j;
			grad_x_sub = ((float*)subImg_x.data)[temp1_sub];
			grad_y_sub = ((float*)subImg_y.data)[temp1_sub];
			mag_sub = grad_y_sub*grad_y_sub + grad_x_sub*grad_x_sub;
			mag_sub = sqrt(mag_sub);
			(sub_mag.data)[temp1_sub] =*(int*)&mag_sub;
		}
	}

	for (row_i = 1; row_i < height - 1; row_i++)     
	{
		temp_sub = row_i * width;
		for (col_j = 1; col_j < width - 1; col_j += 1)
		{
			temp1_sub = temp_sub + col_j;
			grad_x_sub = ((float*)gradImg_x.data)[temp1_sub];
			grad_y_sub = ((float*)gradImg_y.data)[temp1_sub];
			mag_sub = grad_y_sub*grad_y_sub + grad_x_sub*grad_x_sub;
			mag_sub = sqrt(mag_sub);
			(gray_mag.data)[temp1_sub] = *(int*)&mag_sub;
		}
	}
	
	int a, b;
	for (int i = 0; i < height - sub_height; i++)
	{
		int temp = i*width;
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			int temp1 = temp + j;
			for (int in_x = 1; in_x < sub_height-1; in_x++)
			{
				int temp2 = temp1 + in_x*width;
				int sub = in_x*sub_height;
				for (int in_y = 1; in_y < sub_width-1; in_y++)
				{
					int temp3 = temp2 + in_y;
					int pix = (sub_mag.data)[sub + in_y] - (gray_mag.data)[temp3];
					total_diff -= (2 * ((pix >> 31) & 1) - 1)*pix;
				}
			}
			if (Min > total_diff) {
				Min = total_diff;
				a = j;
				b = i;
			}
		}
	}
	*x = a;
	*y = b;
	return 1;
}


int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y) {
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return -1;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	if (grayImg.channels() != 1 || subImg.channels() != 1) {
		cout << "channels is warniing." << endl;
		return -1;
	}
	if (sub_width > width || sub_height > height) {
		cout << "subImg too huge" << endl;
		return -1;
	}

	Mat searchImg = grayImg;
	Mat searchsub = subImg;
	int subhist[256],grayhist[256];
	int flag;
	int Min = 0x00ffffff;
	flag=ustc_CalcHist(searchsub,subhist , 256);


	int a, b;
	for (int i = 0; i < height - sub_height; i++)
	{
		int temp = i*width;
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			for (int k = 0; k < 256; ) {
				grayhist[k] = 0; k += 1;
				grayhist[k] = 0; k += 1;
				grayhist[k] = 0; k += 1;
				grayhist[k] = 0; k += 1;
				grayhist[k] = 0; k += 1;
				grayhist[k] = 0; k += 1;
				grayhist[k] = 0; k += 1;
				grayhist[k] = 0; k += 1;
				grayhist[k] = 0; k += 1;
				grayhist[k] = 0; k += 1;
				grayhist[k] = 0; k += 1;
				grayhist[k] = 0; k += 1;
				grayhist[k] = 0; k += 1;
				grayhist[k] = 0; k += 1;
				grayhist[k] = 0; k += 1;
				grayhist[k] = 0; k += 1;
			}
			int temp1 = temp + j;
			for (int in_x = 0; in_x < sub_height; in_x++)
			{
				int temp2 = temp1 + in_x*width;
				for (int in_y = 0; in_y < sub_width; in_y++)
				{
					grayhist[searchImg.data[temp2 + in_y]]+=1;
				}
			}
			for (int k = 0; k < 256; ) {
				int diff = grayhist[k] - subhist[k];
				total_diff -= (2 * ((diff >> 31) & 1) - 1)*diff;
				k += 1;
				diff = grayhist[k] - subhist[k];
				total_diff -= (2 * ((diff >> 31) & 1) - 1)*diff;
				k += 1;
				diff = grayhist[k] - subhist[k];
				total_diff -= (2 * ((diff >> 31) & 1) - 1)*diff;
				k += 1;
				diff = grayhist[k] - subhist[k];
				total_diff -= (2 * ((diff >> 31) & 1) - 1)*diff;
				k += 1;
				diff = grayhist[k] - subhist[k];
				total_diff -= (2 * ((diff >> 31) & 1) - 1)*diff;
				k += 1;
				diff = grayhist[k] - subhist[k];
				total_diff -= (2 * ((diff >> 31) & 1) - 1)*diff;
				k += 1;
				diff = grayhist[k] - subhist[k];
				total_diff -= (2 * ((diff >> 31) & 1) - 1)*diff;
				k += 1;
				diff = grayhist[k] - subhist[k];
				total_diff -= (2 * ((diff >> 31) & 1) - 1)*diff;
				k += 1;
			}
			if (Min > total_diff) {
				Min = total_diff;
				a = j;
				b = i;
			}
		}
	}
	*x = a;
	*y = b;
	return 1;
}
