#include "SubImageMatch.h"

const float cv_LUT[102] = {
	0, 0.0099996664, 0.019997334, 0.029991005, 0.039978687,
	0.049958397, 0.059928156, 0.069885999, 0.079829983, 0.089758173,
	0.099668652, 0.10955953, 0.11942893, 0.12927501, 0.13909595,
	0.14888994, 0.15865526, 0.16839015, 0.17809294, 0.18776195,
	0.19739556, 0.20699219, 0.21655031, 0.22606839, 0.23554498,
	0.24497867, 0.25436807, 0.26371184, 0.27300870, 0.28225741,
	0.29145679, 0.30060568, 0.30970293, 0.31874755, 0.32773849,
	0.33667481, 0.34555557, 0.35437992, 0.36314702, 0.37185606,
	0.38050637, 0.38909724, 0.39762798, 0.40609807, 0.41450688,
	0.42285392, 0.43113875, 0.43936089, 0.44751999, 0.45561564,
	0.46364760, 0.47161558, 0.47951928, 0.48735857, 0.49513325,
	0.50284320, 0.51048833, 0.51806855, 0.52558380, 0.53303409,
	0.54041952, 0.54774004, 0.55499572, 0.56218672, 0.56931317,
	0.57637525, 0.58337301, 0.59030676, 0.59717667, 0.60398299,
	0.61072594, 0.61740589, 0.62402308, 0.63057774, 0.63707036,
	0.64350110, 0.64987046, 0.65617871, 0.66242629, 0.66861355,
	0.67474097, 0.68080884, 0.68681765, 0.69276786, 0.69865984,
	0.70449406, 0.71027100, 0.71599114, 0.72165483, 0.72726268,
	0.73281509, 0.73831260, 0.74375558, 0.74914461, 0.75448018,
	0.75976276, 0.76499283, 0.77017093, 0.77529752, 0.78037310,
	0.78539819, 0.79037325 };

const double cv_LUT_d[102] = {
	0, 0.00999966668666524, 0.0199973339731505, 0.0299910048568779, 0.0399786871232900,
	0.0499583957219428, 0.0599281551212079, 0.0698860016346425, 0.0798299857122373, 0.0897581741899505,
	0.0996686524911620, 0.109559526773944, 0.119428926018338, 0.129275004048143, 0.139095941482071,
	0.148889947609497, 0.158655262186401, 0.168390157147530, 0.178092938231198, 0.187761946513593,
	0.197395559849881, 0.206992194219821, 0.216550304976089, 0.226068387993884, 0.235544980720863,
	0.244978663126864, 0.254368058553266, 0.263711834462266, 0.273008703086711, 0.282257421981491,
	0.291456794477867, 0.300605670042395, 0.309702944542456, 0.318747560420644, 0.327738506780556,
	0.336674819386727, 0.345555580581712, 0.354379919123438, 0.363147009946176, 0.371856073848581,
	0.380506377112365, 0.389097231055278, 0.397627991522129, 0.406098058317616, 0.414506874584786,
	0.422853926132941, 0.431138740718782, 0.439360887284591, 0.447519975157170, 0.455615653211225,
	0.463647609000806, 0.471615567862328, 0.479519291992596, 0.487358579505190, 0.495133263468404,
	0.502843210927861, 0.510488321916776, 0.518068528456721, 0.525583793551610, 0.533034110177490,
	0.540419500270584, 0.547740013715902, 0.554995727338587, 0.562186743900029, 0.569313191100662,
	0.576375220591184, 0.583373006993856, 0.590306746935372, 0.597176658092678, 0.603982978252998,
	0.610725964389209, 0.617405891751573, 0.624023052976757, 0.630577757214935, 0.637070329275684,
	0.643501108793284, 0.649870449411948, 0.656178717991395, 0.662426293833151, 0.668613567927821,
	0.674740942223553, 0.680808828915828, 0.686817649758645, 0.692767835397122, 0.698659824721463,
	0.704494064242218, 0.710271007486686, 0.715991114416300, 0.721654850864761, 0.727262687996690,
	0.732815101786507, 0.738312572517228, 0.743755584298860, 0.749144624606017, 0.754480183834406,
	0.759762754875771, 0.764992832710910, 0.770170914020331, 0.775297496812126, 0.780373080066636,
	0.785398163397448, 0.790373246728302
};

//函数功能：将bgr图像转化成灰度图像
//bgrImg：彩色图，像素排列顺序是bgr
//grayImg：灰度图，单通道
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg){
	uchar *data,*data_gray;
	int i, j;
	int row, col, step, channels,height,width;
	//grayImg = cvCreateImage(cvGetSize(&bgrImg), 8, 1);
	Mat gray_Img(bgrImg.size(), CV_8U, Scalar(0, 255, 0));
	gray_Img.copyTo(grayImg) ;
	data = bgrImg.data;
	data_gray = grayImg.data;
	row = bgrImg.rows;
	col = bgrImg.cols;
	
	if (NULL == bgrImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	
		for (i = 0; i < row; i++) {
	
	
			for (j = 0; j < col; j++){
				data_gray[i*col + j] =        (114*data[(i*col + j)*3] +    \
					                           587*data[(i*col + j)*3 + 1] + \
											   299*data[(i*col + j)*3 + 2])>>10;
			}
		}

#ifdef IMG_SHOW
		namedWindow("GRAY", CV_WINDOW_NORMAL);
		imshow("GRAY", grayImg);
       /* waitKey();*/
#endif
    
		

		return SUB_IMAGE_MATCH_OK;
};



//函数功能：根据灰度图像计算梯度图像
//grayImg：灰度图，单通道
//gradImg_x：水平方向梯度，浮点类型图像，CV32FC1
//gradImg_y：垂直方向梯度，浮点类型图像，CV32FC1
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y){
	uchar *data;
	uchar *data_x, *data_y;
	int i, j;
	int row, col;
	Mat grad_Img_x(grayImg.size(), CV_32FC1);
	grad_Img_x.copyTo(gradImg_x);
	Mat grad_Img_y(grayImg.size(), CV_32FC1);
	grad_Img_y.copyTo(gradImg_y);
	//((float*)img.data)[i * width + j]
	//gradImg_x.setTo(0);
	//gradImg_y.setTo(0);
	data = grayImg.data;
	data_x = gradImg_x.data;
	data_y = gradImg_y.data;
	row = grayImg.rows;
	col = grayImg.cols;

	for (i = 1; i < row - 1; i++) {


		for (j = 1; j < col- 1; j++){

			unsigned int posi = i * col + j;
			unsigned int left_top = (i - 1)*col + (j - 1);
			unsigned int left_down = left_top + col + col;
			int grad_x, grad_y;

			   grad_x =                     (-data[left_top]       /*- 0 * data[(i - 1)*col + (j)]*/ + data[left_top+2] \
				                             - data[left_top + col]*2 /*+0*data[(i    )*col + (j)]*/ +  data[left_top + col + 2]*2 \
											 - data[left_down]     /*+ 0 * data[(i + 1)*col + (j)]*/ + data[left_down+2]);
            
			   grad_y =                     (-data[left_top] - data[left_top + 1]*2 - data[left_top + 2] \
								              /*-2*data[(i    )*col + (j - 1) ] + 0 * data[(i    )*col + (j) ] +2*data[(i    )*col +(j + 1) ] \*/
											  + data[left_down] + data[left_down + 1]*2 + data[left_down + 2]);

			   ((float*)data_x)[posi] = grad_x;
			   ((float*)data_y)[posi] = grad_y;


		}
	}

#ifdef IMG_SHOW
	Mat gradImg_x_8U(row, col, CV_8UC1);
	Mat gradImg_y_8U(row, col, CV_8UC1);
	//为了方便观察，直接取绝对值
	for (int row_i = 0; row_i <row; row_i++)
	{
		for (int col_j = 0; col_j < col; col_j += 1)
		{
			int val = ((float*)gradImg_x.data)[row_i * col + col_j];
			gradImg_x_8U.data[row_i * col + col_j] = abs(val);

			 val = ((float*)gradImg_y.data)[row_i * col + col_j];
			gradImg_y_8U.data[row_i * col + col_j] = abs(val);
		}
	}

	namedWindow("gradImg_x_8U", 0);
	imshow("gradImg_x_8U", gradImg_x_8U);

	namedWindow("gradImg_y_8U", 0);
	imshow("gradImg_y_8U", gradImg_x_8U);
	//waitKey();
#endif

	

	return SUB_IMAGE_MATCH_OK;
}



//函数功能：根据水平和垂直梯度，计算角度和幅值图
//gradImg_x：水平方向梯度，浮点类型图像，CV32FC1
//gradImg_y：垂直方向梯度，浮点类型图像，CV32FC1
//angleImg：角度图，浮点类型图像，CV32FC1
//magImg：幅值图，浮点类型图像，CV32FC1
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg){
	if (NULL == gradImg_x.data || NULL == gradImg_y.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = gradImg_x.cols;
	int height = gradImg_x.rows;

	Mat angle_Img(gradImg_x.size(), CV_32FC1);
	angle_Img.copyTo(angleImg);
	angleImg.setTo(0);

	Mat mag_Img(gradImg_x.size(), CV_32FC1);
	mag_Img.copyTo(magImg);
	magImg.setTo(0);

	//计算角度图
	for (int row_i = 1, height_no2 = height - 1, width_no2 = width - 1; row_i < height_no2; row_i++)
	{
		for (int col_j = 1; col_j < width_no2; col_j++)
		{
			unsigned int posi = row_i * width + col_j;
			float grad_x = ((float*)gradImg_x.data)[posi];
			float grad_y = ((float*)gradImg_y.data)[posi];
			float angle_ori = atan2(grad_y, grad_x);
			float angle = 0;
			//查表法反正切
			if (grad_x > 0 && grad_y > 0){           //   第一象限
				if (grad_x >= grad_y){
					float f = grad_y / grad_x;
					float f_two = f;

					int index = round(f * 100);

					angle = (cv_LUT[index] + (f_two * 100 - index) * (cv_LUT[index + 1] - cv_LUT[index]));
				}
				else{
					float f = grad_x / grad_y;
					float f_two = f;

					int index = round(f * 100);
					angle = (CV_PI / 2 - (cv_LUT[index] + (f_two * 100 - index) * (cv_LUT[index + 1] - cv_LUT[index])));
				}
			}
			else if (grad_x < 0 && grad_y > 0){           //   第二象限
				if (-grad_x >= grad_y){
					float f = -grad_y / grad_x;
					float f_two = f;

					int index = round(f * 100);
					angle = (CV_PI - (cv_LUT[index] + (f_two * 100 - index) * (cv_LUT[index + 1] - cv_LUT[index])));
				}
				else{
					float f = -grad_x / grad_y;
					float f_two = f;

					int index = round(f * 100);
					angle = (CV_PI / 2 + (cv_LUT[index] + (f_two * 100 - index) * (cv_LUT[index + 1] - cv_LUT[index])));
				}
			}

			else if (grad_x < 0 && grad_y < 0){           //   第三象限
				if (grad_x <= grad_y){
					float f = grad_y / grad_x;
					float f_two = f;

					int index = round(f * 100);
					angle = (CV_PI + (cv_LUT[index] + (f_two * 100 - index) * (cv_LUT[index + 1] - cv_LUT[index])));
				}
				else{
					float f = grad_x / grad_y;
					float f_two = f;

					int index = round(f * 100);
					angle = (CV_PI / 2 * 3 - (cv_LUT[index] + (f_two * 100 - index) * (cv_LUT[index + 1] - cv_LUT[index])));
				}
			}

			else if (grad_x > 0 && grad_y < 0){           //   第四象限
				if (grad_x >= -grad_y){
					float f = -grad_y / grad_x;
					float f_two = f;

					int index = round(f * 100);
					angle = (CV_PI * 2 - (cv_LUT[index] + (f_two * 100 - index) * (cv_LUT[index + 1] - cv_LUT[index])));
				}
				else{
					float f = -grad_x / grad_y;
					float f_two = f;

					int index = round(f * 100);
					angle = (CV_PI / 2 * 3 + (cv_LUT[index] + (f_two * 100 - index) * (cv_LUT[index + 1] - cv_LUT[index])));
				}
			}
			//自己找办法优化三角函数速度，并且转化为角度制，规范化到0-360
			((float*)angleImg.data)[posi] = angle;


			//cout << (angle_ori) / CV_PI * 180 << ": " << angle / CV_PI * 180<<"\t";


			//float mag =sqrt(grad_y*grad_y + grad_x*grad_x);
			//开方运算
			int i;
			float x2, y;
			const float threehalfs = 1.5F;

			float number = grad_y*grad_y + grad_x*grad_x;
			x2 = number * 0.5F;
			y = number;
			i = *(int *)&y;
			i = 0x5f375a86 - (i >> 1);
			y = *(float *)&i;
			y = y * (threehalfs - (x2 * y * y));
			y = y * (threehalfs - (x2 * y * y));
			y = y * (threehalfs - (x2 * y * y));
			/*return number*y;*/

			((float*)magImg.data)[posi] = number*y;
		}
	}

	

#ifdef IMG_SHOW
	Mat angleImg_8U(height, width, CV_8UC1);
	Mat magImg_8U(height, width, CV_8UC1);
	//为了方便观察，进行些许变化
	for (int row_i = 0; row_i < height; row_i++)
	{
		for (int col_j = 0; col_j < width; col_j ++ )
		{
			float angle = ((float*)angleImg.data)[row_i * width + col_j];
			angle *= 180 / CV_PI;
			//angle += 180;
			//为了能在8U上显示，缩小到0-180之间
			angle /= 2;
			angleImg_8U.data[row_i * width + col_j] = angle;

			int val = ((float*)magImg.data)[row_i * width + col_j];
			magImg_8U.data[row_i * width + col_j] = abs(val);
		}
	}

	namedWindow("angleImg_x_8U", 0);
	imshow("angleImg_x_8U", angleImg_8U);

	namedWindow("magImg_x_8U", 0);
	imshow("magImg_x_8U", magImg_8U);
	/*waitKey();*/
#endif
	return SUB_IMAGE_MATCH_OK;
}


//函数功能：对灰度图像进行二值化
//grayImg：灰度图，单通道
//binaryImg：二值图，单通道
//th：二值化阈值，高于此值，255，低于此值0
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th){
	if (NULL == grayImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;

	Mat binary_Img(grayImg.size(), CV_8UC1);
	binary_Img.copyTo(binaryImg);
	binaryImg.setTo(0);

	//int th = th;      //100
	for (int row_i = 0; row_i < height; row_i++)
	{
		int temp0 = row_i * width;
		for (int col_j = 0; col_j < width; col_j ++)
		{
			//int pixVal = grayImg.at<uchar>(row_i, col_j);
			int temp1 = temp0 + col_j;
			int pixVal = grayImg.data[temp1];
			uchar dstVal = 0;
			/*dstVal = pixVal > th ? 255 : 0;*/
			dstVal=~(pixVal-th)>>8;
			binaryImg.data[temp1] = dstVal;
		}
	}

#ifdef IMG_SHOW
	namedWindow("binaryImg", 0);
	imshow("binaryImg", binaryImg);
	//waitKey();
#endif

	return SUB_IMAGE_MATCH_OK;
}


//函数功能：对灰度图像计算直方图
//grayImg：灰度图，单通道
//hist：直方图
//hist_len：直方图的亮度等级，直方图数组的长度
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len){
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
		for (int col_j = 0; col_j < width; col_j ++)
		{
			int pixVal = grayImg.data[row_i * width + col_j];
			hist[pixVal]++;
		}
	}

	//for (int i = 0; i < hist_len; i++){
	//	cout << hist[i]<<' ';
	//}
	//cout << ' ' << endl;
	return SUB_IMAGE_MATCH_OK;
}

//////////////////////////////////////////////////////////////////////////
/////////////////////////////////匹配部分/////////////////////////////////


//产生随机100*100的灰色子图
int ustc_rand_gray(Mat grayImg, Mat& subImg){
	if (NULL == grayImg.data )
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	srand((unsigned)time(NULL));
	int width_begin = (rand() % (width-100+1) );
	int height_begin = (rand() % (height-100+1) );
	cout << height_begin << "   ";
	cout << width_begin << endl;

	Mat sub_Img(100,100, CV_8UC1);
	sub_Img.copyTo(subImg);
	subImg.setTo(0);

	for (int col_i=0; col_i < 100; col_i++){
		for (int row_j = 0; row_j < 100; row_j++){

			subImg.data[col_i*100 + row_j] = grayImg.data[(col_i+height_begin)*width + row_j+width_begin];
		}
		
	}

#ifdef IMG_SHOW
	namedWindow("subgrayImg", 0);
	imshow("subgrayImg", subImg);
	//waitKey();
#endif
	return SUB_IMAGE_MATCH_OK;
}

//产生随机100*100的bgr子图
int ustc_rand_bgr(Mat colorImg, Mat& subImg){
	if (NULL == colorImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = colorImg.cols;
	int height = colorImg.rows;
	srand((unsigned)time(NULL));
	int width_begin = (rand() % (width - 100 + 1));
	int height_begin = (rand() % (height - 100 + 1));
	cout << height_begin << "   ";
	cout << width_begin << endl;

	//width_begin = height_begin = 0;
	Mat sub_Img(100, 100, CV_8UC3);
	sub_Img.copyTo(subImg);
	/*subImg.setTo(0);*/

	for (int col_i = 0; col_i < 100; col_i++){
		for (int row_j = 0; row_j < 100; row_j++){

			int temp = 3*(col_i * 100 + row_j);
			int temp1 = 3*((col_i + height_begin)*width + row_j + width_begin);
			subImg.data[temp] = colorImg.data[temp1];
			subImg.data[temp+1] = colorImg.data[temp1+1];
			subImg.data[temp+2] = colorImg.data[temp1+2];

		}

	}

#ifdef IMG_SHOW
	namedWindow("subcolorImg", 0);
	imshow("subcolorImg", subImg);
	//waitKey();
#endif
	return SUB_IMAGE_MATCH_OK;
}




//函数功能：利用亮度进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y){
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	int min_data = 0x01111111;
	//该图用于记录每一个像素位置的匹配误差
	//Mat searchImg(height, width, CV_32FC1);
	////匹配误差初始化
	//searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int x_i = 0; x_i < sub_height; x_i++)
			{
				uchar *graypixel = grayImg.ptr<uchar>(i + x_i);
				uchar *subpixel = subImg.ptr<uchar>(x_i);
				for (int y_j = 0; y_j < sub_width; y_j++)
				{
					////大图上的像素位置
					////int row_index = i + x_i;
					////int col_index = j + y_j;
					//int bigImg_pix = graypixel[j + y_j];
					////模板图上的像素
					//int template_pix = subpixel[y_j];

					int sub = graypixel[j + y_j] - subpixel[y_j];
					//int abs_sub = sub;
					////int取绝对值
					//sub >>= 31;
					//abs_sub = abs_sub^sub;
					//abs_sub -= sub;

					//total_diff += abs_sub;
					total_diff += sub&0x80000000 ? -sub : sub;
				}
			}
			//存储当前像素位置的匹配误差
			//((float*)searchImg.data)[i * width + j] = total_diff;
			if (min_data > total_diff){
				min_data = total_diff;
				*x = i;
				*y = j;
			}
		}
	}

	

#ifdef IMG_SHOW
    cout << "min place:" << *x << "  " << *y << endl;
	Mat search_gray_8U(height, width, CV_8UC1);
	grayImg.copyTo(search_gray_8U);
	for (int i = 0; i <= 100 ; i++)
	{
		search_gray_8U.data[(*x)*width +(*y)+ i] = 0;
		search_gray_8U.data[(*x + 100)*width + (*y) + i] = 0;
		search_gray_8U.data[(*x+i)*width + (*y)] = 0;
		search_gray_8U.data[(*x + i)*width + (*y)+100] = 0;
		
	}
	namedWindow("search_gray_8U", 0);
	imshow("search_gray_8U", search_gray_8U);
	//waitKey();
#endif
	return SUB_IMAGE_MATCH_OK;
}




//函数功能：利用色彩进行子图匹配
//colorImg：彩色图，三通单
//subImg：模板子图，三通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y){
	if (NULL == colorImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	int min_data = 0x01111111;
	////该图用于记录每一个像素位置的匹配误差
	//Mat searchImg(height, width, CV_32FC1);
	////匹配误差初始化
	//searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int x_i = 0; x_i < sub_height; x_i++)
			{
				for (int y_j = 0; y_j < sub_width; y_j++)
				{
					//大图上的像素位置
					int row_index = i + x_i;
					int col_index = j + y_j;
					int big_pos = 3*(row_index * width + col_index);
					int bigImg_pix_b = colorImg.data[ (big_pos)];
					int bigImg_pix_g = colorImg.data[ (big_pos)+1];
					int bigImg_pix_r = colorImg.data[ (big_pos)+2];
					//模板图上的像素
					int sub_pos =3*( x_i * sub_width + y_j);
					int template_pix_b = subImg.data[ sub_pos];
					int template_pix_g = subImg.data[sub_pos + 1];
					int template_pix_r = subImg.data[ sub_pos + 2];

					int sub_b = bigImg_pix_b - template_pix_b;
					int sub_g = bigImg_pix_g - template_pix_g;
					int sub_r = bigImg_pix_r - template_pix_r;



					total_diff += sub_b & 0x80000000 ? -sub_b : sub_b;
					total_diff += sub_g & 0x80000000 ? -sub_g : sub_g;
					total_diff += sub_r & 0x80000000 ? -sub_r : sub_r;
					//int abs_sub_b = sub_b;
					//int abs_sub_g = sub_g;
					//int abs_sub_r = sub_r;
					////int取绝对值
					//sub_b >>= 31;
					//abs_sub_b = abs_sub_b^sub_b;
					//abs_sub_b -= sub_b;

					//sub_g >>= 31;
					//abs_sub_g = abs_sub_g^sub_g;
					//abs_sub_g -= sub_g;

					//sub_r >>= 31;
					//abs_sub_r = abs_sub_r^sub_r;
					//abs_sub_r -= sub_r;


					//total_diff += abs_sub_b + abs_sub_g + abs_sub_r;
				}
			}
			//存储当前像素位置的匹配误差
			//((float*)searchImg.data)[i * width + j] = total_diff;
			if (0x80000000 & (total_diff-min_data)){
				min_data = total_diff;
				*x = i;
				*y = j;
			}
		}
	}

	

#ifdef IMG_SHOW
    cout << "min place:" << *x << "  " << *y << endl;
	Mat search_color_8U(height, width, CV_8UC3);
	colorImg.copyTo(search_color_8U);
	for (int i = 0; i <= 100; i++)
	{
		search_color_8U.data[3*((*x)*width + (*y) + i)] = 0;
		search_color_8U.data[3 * ((*x)*width + (*y) + i)+1] = 0;
		search_color_8U.data[3 * ((*x)*width + (*y) + i)+2] = 255;

		search_color_8U.data[3*((*x + 100)*width + (*y) + i)] = 0;
		search_color_8U.data[3 * ((*x + 100)*width + (*y) + i)+1] = 0;
		search_color_8U.data[3 * ((*x + 100)*width + (*y) + i)+2] = 0255;

		search_color_8U.data[3*((*x + i)*width + (*y))] = 0;
		search_color_8U.data[3 * ((*x + i)*width + (*y))+1] = 0;
		search_color_8U.data[3 * ((*x + i)*width + (*y))+2] = 0255;

		search_color_8U.data[3*((*x + i)*width + (*y) + 100)] = 0;
		search_color_8U.data[3 * ((*x + i)*width + (*y) + 100)+1] = 0;
		search_color_8U.data[3 * ((*x + i)*width + (*y) + 100)+2] = 0255;

	}
	namedWindow("search_color_8U", 0);
	imshow("search_color_8U", search_color_8U);
	//waitKey();
#endif
	return SUB_IMAGE_MATCH_OK;
}


//函数功能：利用亮度相关性进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y){
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	*x = *y = 0;

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	float max_data = 0;


    unsigned   long long sub_sum = 0;
	int sub_scale = sub_height * sub_width;
	for (int i = 0; i < sub_scale; i++)
	{
		    uchar subpixel_col = subImg.data[i];
			sub_sum += (subpixel_col * subpixel_col);            // (subImg.data[x_i * sub_width + y_j] * subImg.data[x_i * sub_width + y_j]);
		
	}
	
	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < height - sub_height; i++)
	{
		for (int j = 0; j < width - sub_width; j++)
		{
			int ppcc = 0;
			unsigned   long long big_sum = 0;	
			unsigned   long long sub_big_sum = 0;

			//unsigned   long  big_sum = 0;
			//unsigned   long  sub_sum = 0;
			//unsigned   long  sub_big_sum = 0;

			//遍历大图上的每一个像素,计算均值
			for (int x_i = 0; x_i < sub_height; x_i++)
			{
				uchar *graypixel = grayImg.ptr<uchar>(i + x_i);
				uchar *subpixel = subImg.ptr<uchar>(x_i);
				for (int y_j = 0; y_j < sub_width; y_j++)
				{
					//大图上的像素位置
					//int row_index = i + x_i;
					//int col_index = j + y_j;
					uchar graypixel_col = graypixel[j+y_j];
					big_sum += (graypixel_col*graypixel_col);     // (grayImg.data[row_index * width + col_index] * grayImg.data[row_index * width + col_index]);
					//模板图上的像素
					/*uchar subpixel_col = subpixel[y_j];*/
					//sub_sum += (subpixel_col * subpixel_col);            // (subImg.data[x_i * sub_width + y_j] * subImg.data[x_i * sub_width + y_j]);
					sub_big_sum += (graypixel_col * subpixel[y_j]);        //(grayImg.data[row_index * width + col_index] * subImg.data[x_i * sub_width + y_j]);
				}
			}

			//ppcc = sub_big_sum*sub_big_sum*10  / (big_sum*sub_sum);
			//cout << ppcc << " ";
			if (sub_big_sum*sub_big_sum == (big_sum*sub_sum)){
				/*max_data = ppcc;*/
				//cout << ppcc << " " ;
				*x = i;
				*y = j;
			}
		}
	}

	if (*x ==0 ==*y ){
		cout << "no match" << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}
	

#ifdef IMG_SHOW
    cout <<  endl;
	cout << "min place:" << *x << "  " << *y << endl;
	Mat match_var_8U(height, width, CV_8UC1);
	grayImg.copyTo(match_var_8U);
	for (int i = 0; i <= 100; i++)
	{
		match_var_8U.data[(*x)*width + (*y) + i] = 0;
		match_var_8U.data[(*x + 100)*width + (*y) + i] = 0;
		match_var_8U.data[(*x + i)*width + (*y)] = 0;
		match_var_8U.data[(*x + i)*width + (*y) + 100] = 0;

	}
	namedWindow("match_var_8U", 0);
	imshow("match_var_8U", match_var_8U);
	//waitKey();
#endif
	return SUB_IMAGE_MATCH_OK;
}


//函数功能：利用角度值进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y){
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	Mat gradImg_gray_x, gradImg_gray_y, angleImg_gray, magImg_gray;
	Mat gradImg_sub_x, gradImg_sub_y, angleImg_sub, magImg_sub;

	ustc_CalcGrad( grayImg, gradImg_gray_x, gradImg_gray_y);
	ustc_CalcAngleMag(gradImg_gray_x, gradImg_gray_y, angleImg_gray, magImg_gray);

	ustc_CalcGrad(subImg, gradImg_sub_x, gradImg_sub_y);
	ustc_CalcAngleMag(gradImg_sub_x, gradImg_sub_y, angleImg_sub, magImg_sub);
	//destroyAllWindows();

	*x = *y = 0;

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	float min_data = FLT_MAX;
	////该图用于记录每一个像素位置的匹配误差
	//Mat searchImg(height, width, CV_32FC1);
	////匹配误差初始化
	//searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0, height_no2 = height - sub_height, width_no2 = width - sub_width; i <height_no2; i++)
	{
		for (int j = 0; j < width_no2; j++)
		{
			float total_diff = 0;
			//遍历模板图上的每一个像素
			for (int x_i = 1; x_i < sub_height-1; x_i++)
			{
				for (int y_j = 1; y_j < sub_width-1; y_j++)
				{
					//大图上的像素位置
					int row_index = i + x_i;
					int col_index = j + y_j;

					float bigImg_angle = ((float*)angleImg_gray.data)[row_index*width + col_index];
					float template_angle = ((float*)angleImg_sub.data)[x_i*sub_width + y_j];

					float sub = bigImg_angle - template_angle;
					(*((int *)&sub)) &= 0x7fffffff;
					total_diff += sub;
				}
			}
			////存储当前像素位置的匹配误差
			//((float*)searchImg.data)[i * width + j] = total_diff;
			if (min_data > total_diff){
				min_data = total_diff;
				*x = i;
				*y = j;
			}
		}
	}

	

#ifdef IMG_SHOW
    cout << "min place:" << *x << "  " << *y << endl;
	Mat search_gray_8U(height, width, CV_8UC1);
	grayImg.copyTo(search_gray_8U);
	for (int i = 0; i <= 100; i++)
	{
		search_gray_8U.data[(*x)*width + (*y) + i] = 0;
		search_gray_8U.data[(*x + 100)*width + (*y) + i] = 0;
		search_gray_8U.data[(*x + i)*width + (*y)] = 0;
		search_gray_8U.data[(*x + i)*width + (*y) + 100] = 0;

	}
	namedWindow("match_angle_8U", 0);
	imshow("match_angle_8U", search_gray_8U);
	//waitKey();
#endif
	return SUB_IMAGE_MATCH_OK;
}


//函数功能：利用幅值进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y){
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	Mat gradImg_gray_x, gradImg_gray_y, angleImg_gray, magImg_gray;
	Mat gradImg_sub_x, gradImg_sub_y, angleImg_sub, magImg_sub;

	ustc_CalcGrad(grayImg, gradImg_gray_x, gradImg_gray_y);
	ustc_CalcAngleMag(gradImg_gray_x, gradImg_gray_y, angleImg_gray, magImg_gray);

	ustc_CalcGrad(subImg, gradImg_sub_x, gradImg_sub_y);
	ustc_CalcAngleMag(gradImg_sub_x, gradImg_sub_y, angleImg_sub, magImg_sub);
	//destroyAllWindows();

	*x = *y = 0;

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	int min_data = 0x01111111;
	////该图用于记录每一个像素位置的匹配误差
	//Mat searchImg(height, width, CV_32FC1);
	////匹配误差初始化
	//searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0, height_no2 = height - sub_height, width_no2 = width - sub_width; i <height_no2; i++)
	{
		for (int j = 0; j < width_no2; j++)
		{
			float total_diff = 0;
			//遍历模板图上的每一个像素
			for (int x_i = 1; x_i < sub_height - 1; x_i++)
			{
				for (int y_j = 1; y_j < sub_width - 1; y_j++)
				{
					//大图上的像素位置
					int row_index = i + x_i;
					int col_index = j + y_j;
					//大图上的角度
					//int bigImg_angle =atan2( (grayImg.data[(col_index - 1) * width + row_index + 1]
					//					+ 2 * grayImg.data[(col_index)* width + row_index + 1]
					//					+ grayImg.data[(col_index + 1)* width + row_index + 1]
					//					- grayImg.data[(col_index - 1) * width + row_index - 1]
					//					- 2 * grayImg.data[(col_index)* width + row_index - 1]
					//					- grayImg.data[(col_index + 1)* width + row_index - 1])
					//					,
					//				( grayImg.data[(row_index - 1) * width + col_index + 1]
					//	            + 2 * grayImg.data[(row_index)* width + col_index + 1]
					//				+ grayImg.data[(row_index + 1)* width + col_index + 1]
					//				- grayImg.data[(row_index - 1) * width + col_index - 1]
					//				- 2 * grayImg.data[(row_index)* width + col_index - 1]
					//				- grayImg.data[(row_index + 1)* width + col_index - 1]));   //atan2(grad_y, grad_x)
					////模板图上的角度
					//int template_angle = atan2((subImg.data[(col_index - 1) * width + row_index + 1]
					//						+ 2 * subImg.data[(col_index)* width + row_index + 1]
					//						+ subImg.data[(col_index + 1)* width + row_index + 1]
					//						- subImg.data[(col_index - 1) * width + row_index - 1]
					//						- 2 * subImg.data[(col_index)* width + row_index - 1]
					//						- subImg.data[(col_index + 1)* width + row_index - 1])
					//						,
					//						(subImg.data[(row_index - 1) * width + col_index + 1]
					//						+ 2 * subImg.data[(row_index)* width + col_index + 1]
					//						+ subImg.data[(row_index + 1)* width + col_index + 1]
					//						- subImg.data[(row_index - 1) * width + col_index - 1]
					//						- 2 * subImg.data[(row_index)* width + col_index - 1]
					//						- subImg.data[(row_index + 1)* width + col_index - 1]));

					float bigImg_angle = ((float*)magImg_gray.data)[row_index*width + col_index];
					float template_angle = ((float*)magImg_sub.data)[x_i * 100 + y_j];

					float sub = bigImg_angle - template_angle;
					(*((int *)&sub)) &= 0x7fffffff;
					//sub += (float)(3 << 21);
					//int int_sub = (*((int*)&sub) - 0x4ac00000) >> 1;

					total_diff += sub;
				}
			}
			//存储当前像素位置的匹配误差
			//((float*)searchImg.data)[i * width + j] = total_diff;
			if (min_data > total_diff){
				min_data = total_diff;
				*x = i;
				*y = j;
			}
		}
	}

	

#ifdef IMG_SHOW
    cout << "min place:" << *x << "  " << *y << endl;
	Mat search_gray_8U(height, width, CV_8UC1);
	grayImg.copyTo(search_gray_8U);
	for (int i = 0; i <= 100; i++)
	{
		search_gray_8U.data[(*x)*width + (*y) + i] = 0;
		search_gray_8U.data[(*x + 100)*width + (*y) + i] = 0;
		search_gray_8U.data[(*x + i)*width + (*y)] = 0;
		search_gray_8U.data[(*x + i)*width + (*y) + 100] = 0;

	}
	namedWindow("match_angle_8U", 0);
	imshow("match_angle_8U", search_gray_8U);
	//waitKey();
#endif
	return SUB_IMAGE_MATCH_OK;
}



//函数功能：利用直方图进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y){
	if (NULL == grayImg.data || NULL == subImg.data)
	{
		cout << "image is NULL." << endl;
		return SUB_IMAGE_MATCH_FAIL;
	}

	int sub_hist[256];
	ustc_CalcHist(subImg, sub_hist,256);

	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;

	int hist_temp[256] = {0};

	int min_data = 9999999;

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0, height_no2 = height - sub_height, width_no2 = width - sub_width; i <height_no2; i++)
	{
		for (int j = 0; j < width_no2; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int x_i = 0; x_i < sub_height; x_i++)
			{
                uchar *graypixel = grayImg.ptr<uchar>(i + x_i);
				for (int y_j = 0; y_j < sub_width; y_j++)
				{
						uchar bigImg_pix = graypixel[j + y_j];
						hist_temp[bigImg_pix]++;
					
				 }
			}

			for (int count = 0; count<256; count++){
				int sub = hist_temp[count] - sub_hist[count];
				total_diff += sub>0 ? sub : -sub;
				hist_temp[count] = 0;
				
			}
			//存储当前像素位置的匹配误差
			if (min_data > total_diff){
				min_data = total_diff;
				*x = i;
				*y = j;
			}
		}
	}
	

#ifdef IMG_SHOW
    cout << "min place:" << *x << "  " << *y << endl;
	Mat search_gray_8U(height, width, CV_8UC1);
	grayImg.copyTo(search_gray_8U);
	for (int i = 0; i <= 100; i++)
	{
		search_gray_8U.data[(*x)*width + (*y) + i] = 0;
		search_gray_8U.data[(*x + 100)*width + (*y) + i] = 0;
		search_gray_8U.data[(*x + i)*width + (*y)] = 0;
		search_gray_8U.data[(*x + i)*width + (*y) + 100] = 0;

	}
	namedWindow("search_gray_8U", 0);
	imshow("search_gray_8U", search_gray_8U);
	//waitKey();
#endif
	return SUB_IMAGE_MATCH_OK;
}
