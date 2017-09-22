#include "SubImageMatch.h"

int ustc_ConvertBgr2Gray(Mat bgrImg,Mat& grayImg) 
{
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = bgrImg.cols;
	int height = bgrImg.rows;
	for (int row_i = 0; row_i<height; row_i+=1)
	{
		for (int col_j = 0;col_j<width; col_j +=2)
		{
			int mul= 3 * (row_i * width + col_j);
			int b1 = bgrImg.data[mul + 0];
			int g1= bgrImg.data[mul + 1];
			int r1= bgrImg.data[mul + 2];
			int b2 = bgrImg.data[mul + 3];
			int g2 = bgrImg.data[mul + 4];
			int r2 = bgrImg.data[mul + 5];

			int grayVal1 =( b1 *19595 + g1 * 38469 + r1 *7472)>>16;
			int grayVal2 =(b2 * 19595 + g2* 38469 + r2* 7472) >> 16;
			grayImg.data[mul/3] = grayVal1;
			grayImg.data[mul/3+1] = grayVal2;
		}
	}

#ifdef IMG_SHOW
	namedWindow("grayImg", 0);
	imshow("grayImg", grayImg);
	waitKey(0);
#endif
	return SUB_IMAGE_MATCH_OK;
}


int ustc_CalcGrad(Mat grayImg,Mat &gradImg_x,Mat &gradImg_y)
{
		if (NULL == grayImg.data)
		{
			cout << "image is NULL." << endl;
			return SUB_IMAGE_MATCH_FAIL;
		}

		int width = grayImg.cols;
		int height = grayImg.rows;


		//计算x方向梯度图
		for (int row_i = 1; row_i < height-1; row_i+=1)
		{
			for (int col_j = 1; col_j < width-1; col_j += 1)
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
        
		//计算y方向梯度图
		for (int row_i = 1;row_i < height - 1;row_i+=1)
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

#ifdef OK
		Mat gradImg_x_8U(height, width, CV_8UC1);
		gradImg_x_8U.setTo(0);


		//为了方便观察，直接取绝对值
		for (int row_i = 0; row_i < height; row_i++)
		{
			for (int col_j = 0; col_j < width; col_j += 1)
			{
				float a = ((float*)gradImg_x.data)[row_i * width + col_j];
				float b = ((float*)gradImg_y.data)[row_i * width + col_j];
				float val = abs(b);

				gradImg_x_8U.data[row_i * width + col_j] = val;
			}
		}

		namedWindow("aaa", 0);
		imshow("aaa", gradImg_x_8U);
		waitKey(0);
#endif
	return SUB_IMAGE_MATCH_OK;
}




int ustc_CalcAngleMag(Mat gradImg_x,Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
	float ustc_arctan[420]{ 0,0.0996687,0.197396,0.291457,0.380506,0.463648,0.54042,0.610726,0.674741,0.732815,0.785398,0.832981,0.876058,0.915101,0.950547,0.982794,1.0122,1.03907,1.0637,1.08632,1.10715,1.12638,1.14417,1.16067,1.17601,1.19029,1.20362,1.21609,1.22777,1.23874,1.24905,1.25875,1.26791,1.27656,1.28474,1.2925,1.29985,1.30683,1.31347,1.31979,1.32582,1.33156,1.33705,1.3423,1.34732,1.35213,1.35674,1.36116,1.3654,1.36948,1.3734,1.37717,1.38081,1.38431,1.38769,1.39094,1.39409,1.39713,1.40006,1.4029,1.40565,1.40831,1.41088,1.41338,1.4158,1.41815,1.42042,1.42264,1.42478,1.42687,1.4289,1.43087,1.43279,1.43466,1.43647,1.43824,1.43997,1.44165,1.44329,1.44488,1.44644,1.44796,1.44944,1.45089,1.45231,1.45369,1.45504,1.45636,1.45765,1.45891,1.46014,1.46135,1.46253,1.46368,1.46481,1.46592,1.467,1.46807,1.46911,1.47013,1.47113,1.47211,1.47307,1.47401,1.47494,1.47584,1.47674,1.47761,1.47847,1.47931,1.48014,1.48095,1.48175,1.48253,1.4833,1.48406,1.4848,1.48553,1.48625,1.48696,1.48766,1.48834,1.48901,1.48967,1.49033,1.49097,1.4916,1.49222,1.49283,1.49343,1.49402,1.49461,1.49518,1.49575,1.49631,1.49686,1.4974,1.49793,1.49846,1.49898,1.49949,1.49999,1.50049,1.50098,1.50146,1.50194,1.50241,1.50287,1.50333,1.50378,1.50423,1.50467,1.5051,1.50553,1.50595,1.50637,1.50678,1.50719,1.50759,1.50799,1.50838,1.50876,1.50915,1.50952,1.5099,1.51026,1.51063,1.51099,1.51134,1.51169,1.51204,1.51238,1.51272,1.51306,1.51339,1.51372,1.51404,1.51436,1.51468,1.51499,1.5153,1.5156,1.51591,1.51621,1.5165,1.51679,1.51708,1.51737,1.51766,1.51794,1.51821,1.51849,1.51876,1.51903,1.5193,1.51956,1.51982,1.52008,1.52033,1.52059,1.52084,1.52109,1.52133,1.52158,1.52182,1.52205,1.52229,1.52252,1.52276,1.52299,1.52321,1.52344,1.52366,1.52388,1.5241,1.52432,1.52453,1.52475,1.52496,1.52517,1.52537,1.52558,1.52578,1.52598,1.52618,1.52638,1.52658,1.52677,1.52696,1.52716,1.52735,1.52753,1.52772,1.5279,1.52809,1.52827,1.52845,1.52863,1.5288,1.52898,1.52915,1.52933,1.5295,1.52967,1.52984,1.53,1.53017,1.53033,1.5305,1.53066,1.53082,1.53098,1.53113,1.53129,1.53145,1.5316,1.53175,1.53191,1.53206,1.53221,1.53235,1.5325,1.53265,1.53279,1.53294,1.53308,1.53322,1.53336,1.5335,1.53364,1.53378,1.53391,1.53405,1.53418,1.53432,1.53445,1.53458,1.53471,1.53484,1.53497,1.5351,1.53522,1.53535,1.53548,1.5356,1.53572,1.53585,1.53597,1.53609,1.53621,1.53633,1.53645,1.53656,1.53668,1.5368,1.53691,1.53703,1.53714,1.53725,1.53736,1.53748,1.53759,1.5377,1.53781,1.53791,1.53802,1.53813,1.53823,1.53834,1.53845,1.53855,1.53865,1.53876,1.53886,1.53896,1.53906,1.53916,1.53926,1.53936,1.53946,1.53956,1.53965,1.53975,1.53985,1.53994,1.54004,1.54013,1.54022,1.54032,1.54041,1.5405,1.54059,1.54069,1.54078,1.54087,1.54095,1.54104,1.54113,1.54122,1.54131,1.54139,1.54148,1.54156,1.54165,1.54173,1.54182,1.5419,1.54199,1.54207,1.54215,1.54223,1.54231,1.54239,1.54248,1.54256,1.54263,1.54271,1.54279,1.54287,1.54295,1.54303,1.5431,1.54318,1.54326,1.54333,1.54341,1.54348,1.54356,1.54363,1.5437,1.54378,1.54385,1.54392,1.54399,1.54406,1.54414,1.54421,1.54428,1.54435,1.54442,1.54449,1.54456,1.54462,1.54469,1.54476,1.54483,1.5449,1.54496,1.54503,1.5451,1.54516,1.54523,1.54529,1.54536,1.54542,1.54549,1.54555,1.54561,1.54568,1.54574,1.5458 };
	if (NULL == gradImg_x.data|| NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = gradImg_x.cols;
	int height = gradImg_x.rows;


	//计算角度图
	
	for (int row_i = 1; row_i < height - 1; row_i++)
	{
		for (int col_j = 1; col_j < width - 1; col_j += 1)
		{
			float grad_x = ((float*)gradImg_x.data)[row_i * width + col_j];
			float grad_y = ((float*)gradImg_y.data)[row_i * width + col_j];
			float bizhi = grad_y / grad_x;
			float angle=0;
			//angle=atan2(grad_y,grad_x);

			//arctan(dy/dx)
			if (grad_x == 0&&grad_y)angle = 0;
			else if (grad_x&&grad_y == 0)angle = 1.57;
			else if (grad_x > 0) {
			 if (bizhi >= 0 && bizhi < 40)
			   {
				int num = (int)(bizhi * 10) % 400;
				angle = ustc_arctan[num];
 			   }
 			    else if (bizhi <= 0 && bizhi > -40) {
 				int num = (int)(-bizhi * 10) % 400;
 				angle = ustc_arctan[num];
 	 		 }
 			 else if (bizhi >= 40)angle = 1.57;
			 else if (bizhi <= -40)angle = -1.57;
			 }

			else if (grad_x < 0) {
				if (bizhi >= 0 && bizhi < 40)
				{
					int num = (int)(bizhi * 10) % 400;
				angle = ustc_arctan[num];
			}
			else if (bizhi <= 0 && bizhi > -40) {
				int num = (int)(-bizhi * 10) % 400;
				angle = ustc_arctan[num];
			}
			else if (bizhi >= 40)angle = 1.57;
			else if (bizhi <= -40)angle = -1.57;
			angle += 3.14;
			}

			//自己找办法优化三角函数速度，并且转化为角度制，规范化到0-360
			((float*)angleImg.data)[row_i * width + col_j] = angle;
		}
	}


//计算幅值图
for (int row_i = 0; row_i < height; row_i++)
		{
			for (int col_j = 0; col_j < width; col_j += 1)
			{
				float a = ((float*)gradImg_x.data)[row_i * width + col_j];
				float b = ((float*)gradImg_y.data)[row_i * width + col_j];
				float val = a*a + b*b;
				int t = *(int*)&val;
				t -= 0x3f800000; t >>= 1;
				t += 0x3f800000;
				
				((float*)magImg.data)[row_i * width + col_j] =(float)t;
	}
}
#ifdef IMG_SHOW
	Mat angleImg_8U(height, width, CV_8UC1);
	//为了方便观察，进行些许变化
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
		    float angle = ((float*)angleImg.data)[row_i * width + col_j];
			angle *= 180 / CV_PI;
			angle += 180;
			angle /=2;
			//为了能在8U上显示，缩小到0-180之间
			angleImg_8U.data[row_i * width + col_j] = angle;
		}
		
	}
	namedWindow("angleImg_8U", 0);
	imshow("angleImg_8U", angleImg_8U);
	waitKey();


	Mat magImg_8U(height, width, CV_8UC1);
	
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j += 1)
		{
			int t = (int)(((float*)magImg.data)[row_i*width+col_j]);

			magImg_8U.data[row_i * width + col_j] = t;
		}
}

	namedWindow("aaa", 0);
	imshow("aaa", magImg_8U);
	waitKey(0);
#endif
	return SUB_IMAGE_MATCH_OK;
}


int ustc_Threshold(Mat grayImg,Mat& binaryImg,int th)
{
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;

	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp0 = row_i * width;

		for (int col_j = 0; col_j < width; col_j++)
		{
			
			int temp1 = temp0 + col_j;
			int pixVal = grayImg.data[temp1];
			int dstVal = (pixVal>th) * 255;
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


int ustc_CalcHist(Mat grayImg,int* hist,int hist_len)
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
			hist[i] = 0;i++;
			hist[i] = 0;i++;
			hist[i] = 0;i++;
			hist[i] = 0;i++;
			hist[i] = 0;i++;
			hist[i] = 0;i++;
			hist[i] = 0;i++;
			hist[i] = 0;i++;
			hist[i] = 0;i++;
			hist[i] = 0;i++;
			hist[i] = 0;i++;
			hist[i] = 0;i++;
			hist[i] = 0;i++;
			hist[i] = 0;i++;
			hist[i] = 0;i++;
			hist[i] = 0;

		}

		//计算直方图
		for (int row_i = 0; row_i < height; row_i++)
		{
			int pix = row_i * width;
			for (int col_j = 0; col_j < width; col_j++)
			{
				int pixVal = grayImg.data[pix + col_j];
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

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int y = 0; y < sub_height; y++)
			{   
				int row_index = i + y;
				int row_index1 = row_index*width;
				for (int x = 0; x < sub_width; x++)
				{
					//大图上的像素位置
					
					int col_index = j + x;
					int bigImg_pix = grayImg.data[row_index1 + col_index];
					//模板图上的像素
					int template_pix = subImg.data[y * sub_width + x];
					int a = bigImg_pix - template_pix;
					if (bigImg_pix - template_pix < 0)a = -a;
					total_diff +=a;
				}
				}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[i * width + j] = total_diff;
		}
	}

	 
	int min= searchImg.data[0];
	int min_height=0, min_width=0;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			if(((float*)searchImg.data)[i * width + j]<=min)
			{
				min = ((float*)searchImg.data)[i * width + j];
				min_height = i;
				min_width = j;
			}
		}
	}
	
	Mat Sub_Img(sub_height,sub_width, CV_8UC1);
	for(int i=0;i<sub_height;i++)
		for (int j = 0;j < sub_width;j++)
		{
			Sub_Img.data[i*sub_width+j] = grayImg.data[  (i+ min_height)*width+ min_width +j];
		}
#ifdef IMG_SHOW
	namedWindow("sub_Img", 0);
	imshow("sub_Img", Sub_Img);
	waitKey(1);
#endif
	*x = min_width;*y = min_height;
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

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diffb = 0;
			int total_diffg = 0;
			int total_diffr = 0;
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int y = 0; y < sub_height; y++)
			{
				for (int x = 0; x < sub_width; x++)
				{
					//大图上的像素位置
					int row_index = i + y;
					int col_index = j + x;
					int mul = 3 * (row_index * width + col_index);
					int  bigImg_b = colorImg.data[mul + 0];
					int  bigImg_g = colorImg.data[mul + 1];
					int  bigImg_r = colorImg.data[mul + 2];
					//模板图上的像素
					int mul1 = 3 * (y * sub_width + x);
					int template_b = subImg.data[mul1 + 0];
					int template_g = subImg.data[mul1 + 1];
					int template_r = subImg.data[mul1 + 2];

					total_diffb += abs(bigImg_b - template_b);
					total_diffg += abs(bigImg_g - template_g);
					total_diffr += abs(bigImg_r - template_r);
					total_diff += (total_diffr + total_diffg + total_diffb);

				}
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[(i * width + j)] = total_diff;
		}
	}

	float min = ((float*)searchImg.data)[0];
	int min_height = 0, min_width = 0;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			if (((float*)searchImg.data)[(i * width + j)] < min)
			{
				min = ((float*)searchImg.data)[i * width + j];
				//cout << max;
				min_height = i;
				min_width = j;
			}
			}
		}
	
	//cout << max_height;
	//cout << max_width;

	Mat Sub_Img(sub_height, sub_width, CV_8UC3);
	for (int i = 0;i<sub_height;i++)
		for (int j = 0;j < sub_width;j++)
		{
			Sub_Img.data[3*(i*sub_width + j)+0] = colorImg.data[3*((i+ min_height)*width + j+ min_width)+0];
			Sub_Img.data[3 * (i*sub_width + j)+1] = colorImg.data[3*((i+ min_height)*width + j+ min_width)+1];
			Sub_Img.data[3 * (i*sub_width + j)+2] = colorImg.data[3*((i+ min_height)*width + j+ min_width)+2];
}
#ifdef IMG_SHOW
	namedWindow("sub_Img", 0);
	imshow("sub_Img", Sub_Img);
	waitKey(1);
#endif

	*x = min_width;*y = min_height;
	return SUB_IMAGE_MATCH_OK;
}



int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_OK;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_st = 0;
			int total_ss = 0;
			int total_tt = 0;
			float total_r = 0;
			//遍历模板图上的每一个像素
			for (int y = 0; y < sub_height; y++)
			{
				for (int x = 0; x < sub_width; x++)
				{
					//大图上的像素位置
					int row_index = i + y;
					int col_index = j + x;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					//模板图上的像素
					int template_pix = subImg.data[y * sub_width + x];
					total_st += bigImg_pix*template_pix;
					total_ss += bigImg_pix*bigImg_pix;
					total_tt += template_pix*template_pix;

				}
			}
			//存储当前像素位置亮度相关性
			int tmp = ((0x3f800000 << 1) + 0x3f800000 - *(long*)&total_ss) >> 1; float y = *(float*)&tmp; float _sqrt1=y * (1.47f - 0.47f * total_ss * y * y);
			int tmp2 = ((0x3f800000 << 1) + 0x3f800000 - *(long*)&total_tt) >> 1; float y2 = *(float*)&tmp2;float _sqrt2= y2 * (1.47f - 0.47f * total_ss * y2 * y2);

			total_r =(float) total_st *_sqrt1*_sqrt2;
			((float*)searchImg.data)[i * width + j] = total_r;
		}
	}

	float max = ((float*)searchImg.data)[0];
	int max_height = 0, max_width = 0;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			if (((float*)searchImg.data)[i * width + j]>max)
			{
				max = ((float*)searchImg.data)[i * width + j];
				//cout << max;
				max_height = i;
				max_width = j;
			}
		}
	}
	//cout << max_height;
	//cout << max_width;

	Mat Sub_Img(sub_height, sub_width, CV_8UC1);
	for (int i = 0;i<sub_height;i++)
		for (int j = 0;j < sub_width;j++)
		{
			Sub_Img.data[i*sub_width + j] = grayImg.data[(i+ max_height)*width + j+ max_width];
		}
#ifdef IMG_SHOW
	namedWindow("sub_Img", 0);
	imshow("sub_Img", Sub_Img);
	waitKey(1);
#endif
	*x = max_width;*y = max_height;
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

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height-1; i++)
	{
		for (int j = 0; j < width - sub_width-1; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int y = 1; y < sub_height-1; y++)
			{
				for (int x = 1; x < sub_width-1; x++)
				{
					//大图上的像素位置
					int row_index = i + y;
					int col_index = j + x;
					
					//计算大图上的像素x梯度
					int bigImg_grad_x =
						grayImg.data[(row_index - 1) * width + col_index + 1]
						+ 2 * grayImg.data[(row_index)* width + col_index + 1]
						+ grayImg.data[(row_index + 1)* width + col_index + 1]
						- grayImg.data[(row_index - 1) * width + col_index - 1]
						- 2 * grayImg.data[(row_index)* width + col_index - 1]
						- grayImg.data[(row_index + 1)* width + col_index - 1];
					//计算大图上的像素y梯度
					int bigImg_grad_y =
						-grayImg.data[(row_index - 1) * width + col_index - 1]
						- 2 * grayImg.data[(row_index - 1)* width + col_index]
						- grayImg.data[(row_index - 1)* width + col_index + 1]
						+ grayImg.data[(row_index + 1) * width + col_index - 1]
						+ 2 * grayImg.data[(row_index + 1)* width + col_index]
						+ grayImg.data[(row_index + 1)* width + col_index + 1];

					float bigImg_angle = atan2(bigImg_grad_y, bigImg_grad_x);

					//计算模板图上的像素x梯度
					int template_grad_x =
						subImg.data[(y - 1) * sub_width + x + 1]
						+ 2 * subImg.data[(y)* sub_width + x + 1]
						+ subImg.data[(y + 1)* sub_width + x + 1]
						- subImg.data[(y - 1) * sub_width + x - 1]
						- 2 * subImg.data[(y)* sub_width + x - 1]
						- subImg.data[(y + 1)* sub_width + x - 1];
					//计算模板图上像素y梯度
					int template_grad_y =
						-subImg.data[(y - 1) * sub_width + x - 1]
						- 2 * subImg.data[(y - 1)* sub_width + x]
						- subImg.data[(y - 1)* sub_width + x + 1]
						+ subImg.data[(y + 1) * sub_width + x - 1]
						+ 2 * subImg.data[(y + 1)* sub_width + x]
						+ subImg.data[(y + 1)* sub_width + x + 1];

					float template_angle = atan2(template_grad_y, template_grad_x);
					total_diff += abs(bigImg_angle - template_angle);
				}
			}
			((float*)searchImg.data)[i * width + j] = total_diff;
		}
	}

	float min = ((float*)searchImg.data)[0];
	int min_height = 0, min_width = 0;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			if (((float*)searchImg.data)[i * width + j]<min)
			{
				min = ((float*)searchImg.data)[i * width + j];
				//cout << max;
				min_height = i;
				min_width = j;
			}
		}
	}
	//cout << max_height;
	//cout << max_width;

	Mat Sub_Img(sub_height, sub_width, CV_8UC1);
	for (int i = 0;i<sub_height;i++)
		for (int j = 0;j < sub_width;j++)
		{
			Sub_Img.data[i*sub_width + j] = grayImg.data[(i+min_height)*width + j+ min_width];
		}
#ifdef IMG_SHOW
	namedWindow("sub_Img", 0);
	imshow("sub_Img", Sub_Img);
	waitKey(1);
#endif
	*x = min_width;*y = min_height;
	return SUB_IMAGE_MATCH_OK;
}




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

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height - 1; i++)
	{
		for (int j = 0; j < width - sub_width - 1; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int y = 1; y < sub_height - 1; y++)
			{
				for (int x = 1; x < sub_width - 1; x++)
				{
					//大图上的像素位置
					int row_index = i + y;
					int col_index = j + x;

					//计算大图上的像素x梯度
					int bigImg_grad_x =
						grayImg.data[(row_index - 1) * width + col_index + 1]
						+ 2 * grayImg.data[(row_index)* width + col_index + 1]
						+ grayImg.data[(row_index + 1)* width + col_index + 1]
						- grayImg.data[(row_index - 1) * width + col_index - 1]
						- 2 * grayImg.data[(row_index)* width + col_index - 1]
						- grayImg.data[(row_index + 1)* width + col_index - 1];
					//计算大图上的像素y梯度
					int bigImg_grad_y =
						-grayImg.data[(row_index - 1) * width + col_index - 1]
						- 2 * grayImg.data[(row_index - 1)* width + col_index]
						- grayImg.data[(row_index - 1)* width + col_index + 1]
						+ grayImg.data[(row_index + 1) * width + col_index - 1]
						+ 2 * grayImg.data[(row_index + 1)* width + col_index]
						+ grayImg.data[(row_index + 1)* width + col_index + 1];

					
					float bigImg_mag = sqrt(bigImg_grad_y*bigImg_grad_y + bigImg_grad_x*bigImg_grad_x);

					//计算模板图上的像素x梯度
					int template_grad_x =
						subImg.data[(y - 1) * sub_width + x + 1]
						+ 2 * subImg.data[(y)* sub_width + x + 1]
						+ subImg.data[(y + 1)* sub_width + x + 1]
						- subImg.data[(y - 1) * sub_width + x - 1]
						- 2 * subImg.data[(y)* sub_width + x - 1]
						- subImg.data[(y + 1)* sub_width + x - 1];
					//计算模板图上像素y梯度
					int template_grad_y =
						-subImg.data[(y - 1) * sub_width + x - 1]
						- 2 * subImg.data[(y - 1)* sub_width + x]
						- subImg.data[(y - 1)* sub_width + x + 1]
						+ subImg.data[(y + 1) * sub_width + x - 1]
						+ 2 * subImg.data[(y + 1)* sub_width + x]
						+ subImg.data[(y + 1)* sub_width + x + 1];

					float template_mag = sqrt(template_grad_y*template_grad_y + template_grad_x*template_grad_x);
					total_diff += abs(bigImg_mag- template_mag);
				}
			}
			((float*)searchImg.data)[i * width + j] = total_diff;
		}
	}

	float min = ((float*)searchImg.data)[0];
	int min_height = 0, min_width = 0;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			if (((float*)searchImg.data)[i * width + j]<min)
			{
				min = ((float*)searchImg.data)[i * width + j];
				//cout << max;
				min_height = i;
				min_width = j;
			}
		}
	}

	Mat Sub_Img(sub_height, sub_width, CV_8UC1);
	for (int i = 0;i<sub_height;i++)
		for (int j = 0;j < sub_width;j++)
		{
			Sub_Img.data[i*sub_width + j] = grayImg.data[(i + min_height)*width + j + min_width];
		}
#ifdef IMG_SHOW
	namedWindow("sub_Img", 0);
	imshow("sub_Img", Sub_Img);
	waitKey(1);
#endif
	*x = min_width;*y = min_height;
	return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int *position_x, int *position_y)
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
	int *big_hist = new int[256];
	int *sub_hist = new int[256];

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(height, width, CV_32FC1);
	//匹配误差初始化
	searchImg.setTo(FLT_MAX);


	//模板直方图清零
	for (int i = 0; i < 256; i++)
	{
		sub_hist[i] = 0;i++;
		sub_hist[i] = 0;i++;
		sub_hist[i] = 0;i++;
		sub_hist[i] = 0;i++;
		sub_hist[i] = 0;i++;
		sub_hist[i] = 0;i++;
		sub_hist[i] = 0;i++;
		sub_hist[i] = 0;i++;
		sub_hist[i] = 0;i++;
		sub_hist[i] = 0;i++;
		sub_hist[i] = 0;i++;
		sub_hist[i] = 0;i++;
		sub_hist[i] = 0;i++;
		sub_hist[i] = 0;i++;
		sub_hist[i] = 0;i++;
		sub_hist[i] = 0;
	}

	//计算模板图直方图
	for (int row_i = 0; row_i < sub_height; row_i++)
	{
		int pix = row_i * sub_width;
		for (int col_j = 0; col_j < sub_width; col_j++)
		{
			int pixVal = subImg.data[pix + col_j];
			sub_hist[pixVal]++;

		}
	}



	for (int i = 0; i < height - sub_height + 1; i++)
	{
		for (int j = 0; j < width - sub_width + 1; j++)
		{
			int total_diff = 0;
			//遍历大图子图上的每一个像素


			//子图直方图清零
			for (int i = 0; i < 256; i++)
			{
				big_hist[i] = 0;i++;
				big_hist[i] = 0;i++;
				big_hist[i] = 0;i++;
				big_hist[i] = 0;i++;
				big_hist[i] = 0;i++;
				big_hist[i] = 0;i++;
				big_hist[i] = 0;i++;
				big_hist[i] = 0;i++;
				big_hist[i] = 0;i++;
				big_hist[i] = 0;i++;
				big_hist[i] = 0;i++;
				big_hist[i] = 0;i++;
				big_hist[i] = 0;i++;
				big_hist[i] = 0;i++;
				big_hist[i] = 0;i++;
				big_hist[i] = 0;
			}


			//计算子图图直方图
			for (int row_i = 0; row_i < sub_height; row_i++)
			{
				
				for (int col_j = 0; col_j < sub_width; col_j ++)
				{
					//大图上的像素位置
					int row_index = i + row_i;
					int col_index = j + col_j;

					int pixVal = grayImg.data[row_index*width +col_index];
					big_hist[pixVal]++;

				}
			}

			for (int num = 0;num < 256;num++)
			{
				int a = big_hist[num] - sub_hist[num];
				a = abs(a);
				total_diff += a;
			}
			((float*)searchImg.data)[i * width + j] = total_diff;
		}
	}

	//查找误差最小值
	float min = ((float*)searchImg.data)[0];
	int min_height = 0, min_width = 0;
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			if (((float*)searchImg.data)[i * width + j]<min)
			{
				min = ((float*)searchImg.data)[i * width + j];
				//cout << max;
				min_height = i;
				min_width = j;
			}
		}
	}
#ifdef IMG_SHOW
	Mat Sub_Img(sub_height, sub_width, CV_8UC1);
	for (int i = 0;i<sub_height;i++)
		for (int j = 0;j < sub_width;j++)
		{
			Sub_Img.data[i*sub_width + j] = grayImg.data[(i + min_height)*width + j + min_width];
		}


	namedWindow("sub_Img", 0);
	imshow("sub_Img", Sub_Img);
	waitKey(1);
#endif
	return SUB_IMAGE_MATCH_OK;
}




