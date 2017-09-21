#include "SubImageMatch.h"
int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg) {
	// Gray = (114 * B + 587 * G + 299 * R ) / 1000;使用这一公式避免了小数计算，速度比较快
	//Mat matGray(img.rows, img.cols, CV_8UC1);
	if (NULL == bgrImg.data) {
		cout << "cannot open the pic" << endl;
		return -1;
	}
	if (bgrImg.channels() != 3) {
		cout << "channels wrong" << endl;
		return -1;
	}
	int gray_cols = grayImg.cols;
	int gray_rows = grayImg.rows;
	for (int idx = 0; idx < gray_rows; idx++)
	{
		//uchar *ptr = img.ptr<uchar>(idx);
		//uchar *newPtr = matGray.ptr<uchar>(idx);
		int need = idx*gray_cols;
		for (int subIdx = 0; subIdx < gray_cols; subIdx++)//计算得到像素值
		{
			int need1 = need + subIdx;
			int need2 = 3 * need1;
			grayImg.data[need1] = (114 * bgrImg.data[need2] + 587 * bgrImg.data[++need2]
				+ 299 * bgrImg.data[++need2]) / 1000;
		}
	}
	return 1;
}

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y) {
	//函数实现
	int grad_i, grad_j;
	int grad_rows = grayImg.rows;
	int grad_cols = grayImg.cols;
	float l1, l2;
	if (NULL == grayImg.data) {
		cout << "cannot open the pic" << endl;
		return -1;
	}
	if (grayImg.channels() != 1) {
		cout << "channels wrong" << endl;
		return -1;
	}
	for (grad_i = 0; grad_i < grad_rows; grad_i++)
	{
		int grad_need0 = grad_i*grad_cols;
		for (grad_j = 0; grad_j < grad_cols; grad_j++) {
			int grad_need10 = grad_need0 + grad_j;
			((float*)gradImg_x.data)[grad_need10] = 0;
			((float*)gradImg_y.data)[grad_need10] = 0;
		}
	}
	for (grad_i = 1; grad_i < grad_rows - 1; grad_i++) //获得GX,GY
	{
		int grad_need = grad_i*grad_cols;
		for (grad_j = 1; grad_j < grad_cols - 1; grad_j++) {
			int grad_need1 = grad_need + grad_j;
			l1 = (float)(grayImg.data[grad_need1 + 1 - grad_cols] - grayImg.data[grad_need1 - 1 - grad_cols] + grayImg.data[grad_need1 + 1] + grayImg.data[grad_need1 + 1] - grayImg.data[grad_need1 - 1] - grayImg.data[grad_need1 - 1] + grayImg.data[grad_need1 + 1 + grad_cols] - grayImg.data[grad_need1 - 1 + grad_cols]);
			((float*)gradImg_x.data)[grad_need1] = l1;
			l2 = (float)(grayImg.data[grad_need1 - 1 - grad_cols] - grayImg.data[grad_need1 - 1 + grad_cols] + grayImg.data[grad_need1 - grad_cols] + grayImg.data[grad_need1 - grad_cols] - grayImg.data[grad_need1 + grad_cols] - grayImg.data[grad_need1 + grad_cols] + grayImg.data[grad_need1 + 1 - grad_cols] - grayImg.data[grad_need1 + 1 + grad_cols]);
			((float*)gradImg_y.data)[grad_need1] = l2;
		}
	}
	return 1;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat&
	angleImg, Mat& magImg) {
	//函数实现
	if (NULL == gradImg_x.data || NULL == gradImg_y.data) {
		cout << "cannot open the pic" << endl;
		return -1;
	}
	float atant[] = { 0,0.01746,0.03492,0.05241,0.06993,0.08749,0.1051,0.12278,0.14054,0.15838,
		0.17633,0.19438,0.21256,0.23087,0.24933,0.26795,0.28675,0.30573,0.32492,
		0.34433, 0.36397, 0.38386,0.40403,0.42447, 0.44523,0.46631,0.48773,0.50953,
		0.53171,0.55431,0.57735, 0.60086, 0.62487, 0.6494, 0.67451, 0.70021,0.72654,
		0.75355,0.78129,0.80978,0.8391,0.86929,0.9004,0.93252,0.96569,1,1.03553,
		1.07237,1.11061,1.15037,1.19175,1.2349, 1.27994,1.32704,1.37638,1.42815,
		1.48256,1.53986,1.60033,1.66428,1.73205, 1.80405,1.88073,1.96261, 2.0503,
		2.14451,2.24604,2.35585, 2.47509,2.60509,2.74748,2.90421,3.07768, 3.27085,
		3.48741,3.73205,4.01078,4.33148,4.70463, 5.14455, 5.67128,6.31375,7.11537,
		8.14435,9.51436,11.43005,14.30067,19.08114,28.63625,57.28996,65535,
		-57.28996,-28.63625, -19.08114,-14.30067,-11.43005,-9.51436,-8.14435,-7.11537,
		-6.31375, -5.67128, -5.14455,-4.70463,-4.33148, -4.01078, -3.73205, -3.48741,
		-3.27085, -3.07768, -2.90421,-2.74748,-2.60509,-2.47509,-2.35585, -2.24604,
		-2.14451, -2.0503, -1.96261,-1.88073, -1.80405, -1.73205,-1.66428,-1.60033,
		-1.53986, -1.48256,-1.42815, -1.37638,-1.32704,-1.27994, -1.2349,-1.19175,
		-1.15037,-1.11061,-1.07237, -1.03553, -1,-0.96569,-0.93252, -0.9004,-0.86929,
		-0.8391,-0.80978,-0.78129,-0.75355,-0.72654, -0.70021, -0.67451,-0.64941,-0.62487,
		-0.60086,-0.57735,-0.55431,-0.53171, -0.50953, -0.48773,-0.46631,-0.44523,-0.42447,
		-0.40403,-0.38386, -0.36397, -0.34433,-0.32492,-0.30573, -0.28675,-0.26795,-0.24933,
		-0.23087,-0.21256,-0.19438,-0.17633,-0.15838,-0.14054,-0.12278,-0.1051,-0.08749,
		-0.06993, -0.05241,-0.03492, -0.01746, 0 };
	int i, j, k;
	float tant;
	float l1, l2;
	int heng = gradImg_x.cols;
	int shu = gradImg_x.rows;
	for (i = 0; i < shu; i++) {
		int need = i*heng;
		for (j = 0; j < heng; j++) {
			int need1 = i*heng + j;
			((float*)angleImg.data)[need1] = 45;
			((float*)magImg.data)[need1] = 0;
		}
	}


	for (i = 1; i < shu - 1; i++)//分类查表计算角度
	{
		int need = i*heng;
		for (j = 1; j < heng - 1; j++) {
			int need1 = i*heng + j;
			((float*)magImg.data)[need1] = sqrt(((float*)gradImg_x.data)[need1] * ((float*)gradImg_x.data)[need1] + ((float*)gradImg_y.data)[need1] * ((float*)gradImg_y.data)[need1]);
			l1 = ((float*)gradImg_y.data)[need1];
			l2 = ((float*)gradImg_x.data)[need1];
			tant = l1 / l2;
			if (l2 == 0) {
				if (l1 >= 0)
					((float*)angleImg.data)[need1] = 90;
				else
					((float*)angleImg.data)[need1] = 270;
				continue;
			}
			else if (l1 >= 0) {
				for (k = 0; k < 180; k++)
					if (tant >= atant[k] && tant <= atant[k + 1]) ((float*)angleImg.data)[need1] = k + (tant - atant[k]) / (atant[k + 1] - atant[k]);
				continue;
			}
			else if (l1 < 0) {
				for (k = 180; k <360; k++)
					if (tant >= atant[k] && tant < atant[k + 1])((float*)angleImg.data)[need1] = k + (tant - atant[k]) / (atant[k + 1] - atant[k]);
				continue;
			}
		}
	}
	return 1;
}

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th) {
	//函数实现
	if (NULL == grayImg.data||grayImg.channels()!=1)
	{
		cout << "image is wrong" << endl;
		return -1;
	}
	int cols = grayImg.cols;
	int rows = grayImg.rows;
	for (int row_i = 0; row_i < rows; row_i++)
	{
		int temp0 = row_i * cols;
		for (int col_j = 0; col_j <cols; col_j++)
		{
			int temp1 = temp0 + col_j;
			int pixVal = grayImg.data[temp1];
			binaryImg.data[temp1] = 255 * ((th - pixVal >> 31) & 1);
		}
	}
	namedWindow("binaryImg", 0);
	imshow("binaryImg", binaryImg);
	waitKey();
	return 1;
}

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len) {
	//函数实现
	if (NULL == grayImg.data || NULL == hist||grayImg.channels()!=1)
	{
		cout << "image is wrong" << endl;
		return -1;
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
		int test = row_i * width;
		for (int col_j = 0; col_j < width; col_j++)
		{
			int test1 = test + col_j;
			int pixVal = grayImg.data[test1];
			hist[pixVal]++;
		}
	}
	return 1;
}
int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y) {
	//函数实现
	if (NULL == grayImg.data || NULL == subImg.data|| grayImg.channels() != 1|| subImg.channels() != 1)
	{
		cout << "image is wrong" << endl;
		return -1;
	}
	if (grayImg.cols < subImg.cols || grayImg.rows < subImg.rows) {
		cout << "size wrong" << endl;
		return -1;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int sub_h2 = height - sub_height;
	int sub_w2 = width - sub_width;
	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(sub_h2, sub_w2, CV_32FC1);
	//匹配误差初始化
	//searchImg.setTo(FLT_MAX);
	//int minsub=2560000;
	int best_x=0, best_y=0;
	//遍历大图每一个像素，注意行列的起始、终止坐标

	for (int i = 0; i <sub_h2; i++)
	{
		for (int j = 0; j <sub_w2; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int y1 = 0; y1 < sub_height; y1++)
			{
				int row_index = i + y1;
				int love = row_index * width;
				int hate = y1 * sub_width;
				for (int x1 = 0; x1 < sub_width; x1++)
				{
					//大图上的像素位置
					int col_index = j + x1;
					int bigImg_pix = grayImg.data[love + col_index];
					//模板图上的像素
					int template_pix = subImg.data[hate + x1];
					int sub_pix = bigImg_pix - template_pix;
					total_diff += abs(sub_pix);
					//printf("%d %d\n", subImg.data[hate + x1], grayImg.data[love + col_index]);
				}
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[i *sub_w2 + j] = total_diff;
			/*if (total_diff < minsub) {
			minsub = total_diff;
			best_x = j;
			best_y = i;
			}*/
		}
	}
	float minsub = ((float*)searchImg.data)[0];
	for (int i = 0; i < sub_h2; i++)
	{
		for (int j = 0; j < sub_w2; j++)
		{
			int a = ((float*)searchImg.data)[i * sub_w2 + j];
			if (a < minsub) {
				minsub = a;
				best_x = j;
				best_y = i;
			}
		}
	}
	*x = best_x;
	*y = best_y;
	//printf("%d   %d", *x,* y);
	return 1;
}
int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y){
	//函数实现
	if (NULL == colorImg.data || NULL == subImg.data||colorImg.channels()!=3||subImg.channels()!=3)
	{
		cout << "image is wrong" << endl;
		return -1;
	}
	if (colorImg.cols < subImg.cols || colorImg.rows < subImg.rows) {
		cout << "size wrong" << endl;
		return -1;
	}
	/*if (colorImg.channels != 3 || subImg.channels != 3) {
	cout << "channels error" << endl;
	return -1;
	}*/
	int width = colorImg.cols;
	int height = colorImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int sub_h2 = height - sub_height;
	int sub_w2 = width - sub_width;
	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg1(sub_h2, sub_w2, CV_32FC1);
	//匹配误差初始化
	//searchImg1.setTo(FLT_MAX);
	//int minsub=2560000;
	int best_x=0, best_y=0;
	//遍历大图每一个像素，注意行列的起始、终止坐标

	for (int i = 0; i <sub_h2; i++)
	{
		for (int j = 0; j <sub_w2; j++)
		{
			int total_diff = 0;
			//遍历模板图上的每一个像素
			for (int y1 = 0; y1 < sub_height; y1++)
			{
				int row_index = i + y1;
				int love = row_index * width;
				int hate = y1 * sub_width;
				for (int x1 = 0; x1 < sub_width; x1++)
				{
					//大图上的像素位置
					int col_index = j + x1;
					int use = 3 * (love + col_index);
					int bigImg_pix_b = colorImg.data[use];
					int bigImg_pix_g = colorImg.data[use + 1];
					int bigImg_pix_r = colorImg.data[use + 2];
					//模板图上的像素
					int use1 = 3 * (hate + x1);
					int template_pix_b = subImg.data[use1];
					int template_pix_g = subImg.data[use1 + 1];
					int template_pix_r = subImg.data[use1 + 2];
					int sub_pix = abs(bigImg_pix_b - template_pix_b) + abs(bigImg_pix_g - template_pix_g) + abs(bigImg_pix_r - template_pix_r);
					total_diff += sub_pix;
					//printf("%d %d\n", subImg.data[hate + x1], grayImg.data[love + col_index]);
				}
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg1.data)[i *sub_w2 + j] = total_diff;
			/*if (total_diff < minsub) {
			minsub = total_diff;
			best_x = j;
			best_y = i;
			}*/
		}
	}
	//printf("%f",)
	float minsub = ((float*)searchImg1.data)[0];
	for (int i = 0; i < sub_h2; i++)
	{
		for (int j = 0; j < sub_w2; j++)
		{
			int a = ((float*)searchImg1.data)[i * sub_w2 + j];
			if (a < minsub) {
				minsub = a;
				best_x = j;
				best_y = i;
			}
		}
	}
	*x = best_x;
	*y = best_y;
	//printf("%d   %d", *x,* y);
	return 1;
}
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y) {
	//函数实现
	if (NULL == grayImg.data || NULL == subImg.data || grayImg.channels() != 1 || subImg.channels() != 1)
	{
		cout << "image is wrong" << endl;
		return -1;
	}
	if (grayImg.cols < subImg.cols || grayImg.rows < subImg.rows) {
		cout << "size wrong" << endl;
		return -1;
	}
	int corr_i, corr_j;
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int sub_h2 = height - sub_height;
	int sub_w2 = width - sub_width;
	Mat searchImg(sub_h2, sub_w2, CV_32FC1);
	int best_x = 0, best_y = 0;
	int need = sub_height*sub_width;
	for (corr_i = 0; corr_i <sub_h2; corr_i++)
	{
		for (corr_j = 0; corr_j <sub_w2; corr_j++)
		{
			//int total_diff = 0;
			//遍历模板图上的每一个像素
			//printf("%d\n", grayImg.data[corr_i*width + corr_j]);
			//float fenzi = 0, fenmu1 = 0, fenmu2 = 0;
			float xy = 0, x = 0, y = 0, xx = 0, yy = 0;
			for (int y1 = 0; y1 < sub_height; y1++)
			{
				int row_index = corr_i + y1;
				int love = row_index * width;
				int hate = y1 * sub_width;
				for (int x1 = 0; x1 < sub_width; x1++)
				{
					//大图上的像素位置
					int col_index = corr_j + x1;
					int bigImg_pix = grayImg.data[love + col_index];
					//模板图上的像素
					int template_pix = subImg.data[hate + x1];
					xy += bigImg_pix*template_pix;
					x += bigImg_pix;
					y += template_pix;
					xx += bigImg_pix*bigImg_pix;
					yy += template_pix*template_pix;
					//fenzi += bigImg_pix*template_pix;
					//fenmu1 += bigImg_pix*bigImg_pix;
					//fenmu2 += template_pix*template_pix;
					//total_diff += abs(sub_pix);
					//printf("%d %d\n", subImg.data[hate + x1], grayImg.data[love + col_index]);
				}
			}
			float fenzi = need*xy - x*y;
			float fenmu1 = sqrt(need*xx - x*x);
			float fenmu2 = sqrt(need*yy - y*y);
			float fenmu = fenmu1*fenmu2;
			float total_corr = fenzi / fenmu;
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[corr_i *sub_w2 + corr_j] = total_corr;
			/*if (total_diff < minsub) {
			minsub = total_diff;
			best_x = j;
			best_y = i;
			}*/
		}
	}
	float maxcorr = ((float*)searchImg.data)[0];
	for (int i = 0; i < sub_h2; i++)
	{
		for (int j = 0; j < sub_w2; j++)
		{
			float a = ((float*)searchImg.data)[i * sub_w2 + j];
			if (abs(a - 1) <abs(maxcorr - 1)) {
				maxcorr = a;
				best_x = j;
				best_y = i;
			}
		}
	}
	*x = best_x;
	*y = best_y;
	//printf("%d   %d", *x,* y);
	return 1;
}
int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
	//函数实现
	float atant[] = { 0,0.01746,0.03492,0.05241,0.06993,0.08749,0.1051,0.12278,0.14054,0.15838,
		0.17633,0.19438,0.21256,0.23087,0.24933,0.26795,0.28675,0.30573,0.32492,
		0.34433, 0.36397, 0.38386,0.40403,0.42447, 0.44523,0.46631,0.48773,0.50953,
		0.53171,0.55431,0.57735, 0.60086, 0.62487, 0.6494, 0.67451, 0.70021,0.72654,
		0.75355,0.78129,0.80978,0.8391,0.86929,0.9004,0.93252,0.96569,1,1.03553,
		1.07237,1.11061,1.15037,1.19175,1.2349, 1.27994,1.32704,1.37638,1.42815,
		1.48256,1.53986,1.60033,1.66428,1.73205, 1.80405,1.88073,1.96261, 2.0503,
		2.14451,2.24604,2.35585, 2.47509,2.60509,2.74748,2.90421,3.07768, 3.27085,
		3.48741,3.73205,4.01078,4.33148,4.70463, 5.14455, 5.67128,6.31375,7.11537,
		8.14435,9.51436,11.43005,14.30067,19.08114,28.63625,57.28996,65535,
		-57.28996,-28.63625, -19.08114,-14.30067,-11.43005,-9.51436,-8.14435,-7.11537,
		-6.31375, -5.67128, -5.14455,-4.70463,-4.33148, -4.01078, -3.73205, -3.48741,
		-3.27085, -3.07768, -2.90421,-2.74748,-2.60509,-2.47509,-2.35585, -2.24604,
		-2.14451, -2.0503, -1.96261,-1.88073, -1.80405, -1.73205,-1.66428,-1.60033,
		-1.53986, -1.48256,-1.42815, -1.37638,-1.32704,-1.27994, -1.2349,-1.19175,
		-1.15037,-1.11061,-1.07237, -1.03553, -1,-0.96569,-0.93252, -0.9004,-0.86929,
		-0.8391,-0.80978,-0.78129,-0.75355,-0.72654, -0.70021, -0.67451,-0.64941,-0.62487,
		-0.60086,-0.57735,-0.55431,-0.53171, -0.50953, -0.48773,-0.46631,-0.44523,-0.42447,
		-0.40403,-0.38386, -0.36397, -0.34433,-0.32492,-0.30573, -0.28675,-0.26795,-0.24933,
		-0.23087,-0.21256,-0.19438,-0.17633,-0.15838,-0.14054,-0.12278,-0.1051,-0.08749,
		-0.06993, -0.05241,-0.03492, -0.01746, 0 };
	if (NULL == grayImg.data || NULL == subImg.data || grayImg.channels() != 1 || subImg.channels() != 1)
	{
		cout << "image is wrong" << endl;
		return -1;
	}
	if (grayImg.cols < subImg.cols || grayImg.rows < subImg.rows) {
		cout << "size wrong" << endl;
		return -1;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int sub_h2 = height - sub_height;
	int sub_w2 = width - sub_width;
	//先计算大图的梯度和角度
	//梯度的计算
	Mat gradImg_x_big(height, width, CV_32FC1);
	Mat gradImg_y_big(height, width, CV_32FC1);
	Mat angleImg_big(height, width, CV_32FC1);
	for (int grad_i = 0; grad_i <height; grad_i++)
	{
		int grad_need0 = grad_i*width;
		for (int grad_j = 0; grad_j <width; grad_j++) {
			int grad_need10 = grad_need0 + grad_j;
			((float*)gradImg_x_big.data)[grad_need10] = 0;
			((float*)gradImg_y_big.data)[grad_need10] = 0;
		}
	}
	for (int grad_i = 1; grad_i < height - 1; grad_i++) //获得GX,GY
	{
		int grad_need = grad_i*width;
		for (int grad_j = 1; grad_j < width - 1; grad_j++) {
			int grad_need1 = grad_need + grad_j;
			float l1 = (float)(grayImg.data[grad_need1 + 1 - width] - grayImg.data[grad_need1 - 1 - width] + grayImg.data[grad_need1 + 1] + grayImg.data[grad_need1 + 1] - grayImg.data[grad_need1 - 1] - grayImg.data[grad_need1 - 1] + grayImg.data[grad_need1 + 1 + width] - grayImg.data[grad_need1 - 1 + width]);
			((float*)gradImg_x_big.data)[grad_need1] = l1;
			float l2 = (float)(grayImg.data[grad_need1 - 1 - width] - grayImg.data[grad_need1 - 1 + width] + grayImg.data[grad_need1 - width] + grayImg.data[grad_need1 - width] - grayImg.data[grad_need1 + width] - grayImg.data[grad_need1 + width] + grayImg.data[grad_need1 + 1 - width] - grayImg.data[grad_need1 + 1 + width]);
			((float*)gradImg_y_big.data)[grad_need1] = l2;
		}
	}
	//角度的计算
	float tant;
	int heng = width;
	int shu = height;
	for (int i = 0; i < shu; i++) {
		int need = i*heng;
		for (int j = 0; j < heng; j++) {
			int need1 = i*heng + j;
			((float*)angleImg_big.data)[need1] = 45;
			//((float*)magImg.data)[need1] = 0;
		}
	}


	for (int i = 1; i < shu - 1; i++)//分类查表计算角度
	{
		int need = i*heng;
		for (int j = 1; j < heng - 1; j++) {
			int need1 = i*heng + j;
			float l1_angle = ((float*)gradImg_y_big.data)[need1];
			float l2_angle = ((float*)gradImg_x_big.data)[need1];
			tant = l1_angle / l2_angle;
			if (l2_angle == 0) {
				if (l1_angle >= 0)
					((float*)angleImg_big.data)[need1] = 90;
				else
					((float*)angleImg_big.data)[need1] = 270;
				continue;
			}
			else if (l1_angle >= 0) {
				for (int k = 0; k < 180; k++)
					if (tant >= atant[k] && tant <= atant[k + 1]) ((float*)angleImg_big.data)[need1] = k + (tant - atant[k]) / (atant[k + 1] - atant[k]);
				continue;
			}
			else if (l1_angle < 0) {
				for (int k = 180; k <360; k++)
					if (tant >= atant[k] && tant < atant[k + 1])((float*)angleImg_big.data)[need1] = k + (tant - atant[k]) / (atant[k + 1] - atant[k]);
				continue;
			}
		}
	}
	//再计算小图的梯度和角度
	//梯度的计算
	Mat gradImg_x_small(sub_height, sub_width, CV_32FC1);
	Mat gradImg_y_small(sub_height, sub_width, CV_32FC1);
	Mat angleImg_small(sub_height, sub_width, CV_32FC1);
	for (int grad_i = 0; grad_i <sub_height; grad_i++)
	{
		int grad_need0 = grad_i*sub_width;
		for (int grad_j = 0; grad_j <sub_width; grad_j++) {
			int grad_need10 = grad_need0 + grad_j;
			((float*)gradImg_x_small.data)[grad_need10] = 0;
			((float*)gradImg_y_small.data)[grad_need10] = 0;
		}
	}
	for (int grad_i = 1; grad_i < sub_height - 1; grad_i++) //获得GX,GY
	{
		int grad_need = grad_i*sub_width;
		for (int grad_j = 1; grad_j < sub_width - 1; grad_j++) {
			int grad_need1 = grad_need + grad_j;
			float l1_s = (float)(subImg.data[grad_need1 + 1 - sub_width] - subImg.data[grad_need1 - 1 - sub_width] + subImg.data[grad_need1 + 1] + subImg.data[grad_need1 + 1] - subImg.data[grad_need1 - 1] - subImg.data[grad_need1 - 1] + subImg.data[grad_need1 + 1 + sub_width] - subImg.data[grad_need1 - 1 + sub_width]);
			((float*)gradImg_x_small.data)[grad_need1] = l1_s;
			float l2_s = (float)(subImg.data[grad_need1 - 1 - sub_width] - subImg.data[grad_need1 - 1 + sub_width] + subImg.data[grad_need1 - sub_width] + subImg.data[grad_need1 - sub_width] - subImg.data[grad_need1 + sub_width] - subImg.data[grad_need1 + sub_width] + subImg.data[grad_need1 + 1 - sub_width] - subImg.data[grad_need1 + 1 + sub_width]);
			((float*)gradImg_y_small.data)[grad_need1] = l2_s;
		}
	}
	//角度的计算
	float tant1;
	heng = sub_width;
	shu = sub_height;
	for (int i = 0; i < shu; i++) {
		int need = i*heng;
		for (int j = 0; j < heng; j++) {
			int need1 = i*heng + j;
			((float*)angleImg_small.data)[need1] = 45;
		}
	}


	for (int i = 1; i < shu - 1; i++)//分类查表计算角度
	{
		int need = i*heng;
		for (int j = 1; j < heng - 1; j++) {
			int need1 = i*heng + j;
			float l1_angles = ((float*)gradImg_y_small.data)[need1];
			float l2_angles = ((float*)gradImg_x_small.data)[need1];
			tant1 = l1_angles / l2_angles;
			if (l2_angles == 0) {
				if (l1_angles >= 0)
					((float*)angleImg_small.data)[need1] = 90;
				else
					((float*)angleImg_small.data)[need1] = 270;
				continue;
			}
			else if (l1_angles >= 0) {
				for (int k = 0; k < 180; k++)
					if (tant1 >= atant[k] && tant1 <= atant[k + 1]) ((float*)angleImg_small.data)[need1] = k + (tant1 - atant[k]) / (atant[k + 1] - atant[k]);
				continue;
			}
			else if (l1_angles < 0) {
				for (int k = 180; k <360; k++)
					if (tant1 >= atant[k] && tant1 < atant[k + 1])((float*)angleImg_small.data)[need1] = k + (tant1 - atant[k]) / (atant[k + 1] - atant[k]);
				continue;
			}
		}
	}
	//大图小图的角度已经计算完毕
	//匹配
	Mat searchImg(sub_h2, sub_w2, CV_32FC1);
	int best_x=0, best_y=0;
	int subh_1 = sub_height - 1;
	int subw_1 = sub_width - 1;
	for (int angle_i = 0; angle_i <sub_h2; angle_i++)
	{
		for (int angle_j = 0; angle_j <sub_w2; angle_j++)
		{
			float total_diff = 0;
			//遍历模板图上的每一个像素
			//float fenzi = 0, fenmu1 = 0, fenmu2 = 0;
			for (int y1 = 1; y1 < subh_1; y1++)
			{
				int row_index = angle_i + y1;
				int love = row_index * width;
				int hate = y1 * sub_width;
				for (int x1 = 1; x1 < subw_1; x1++)
				{
					//大图上的像素位置
					int col_index = angle_j + x1;
					float bigImg_pix = ((float*)angleImg_big.data)[love + col_index];
					//模板图上的像素
					float template_pix = ((float*)angleImg_small.data)[hate + x1];
					float sub_pix = bigImg_pix - template_pix;
					total_diff += abs(sub_pix);
					//printf("%d %d\n", subImg.data[hate + x1], grayImg.data[love + col_index]);
				}
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[angle_i *sub_w2 + angle_j] = total_diff;
		}
	}
	float minsub = ((float*)searchImg.data)[0];
	for (int i = 0; i < sub_h2; i++)
	{
		for (int j = 0; j < sub_w2; j++)
		{
			float a = ((float*)searchImg.data)[i * sub_w2 + j];
			if (a <minsub) {
				minsub = a;
				best_x = j;
				best_y = i;
			}
		}
	}
	*x = best_x;
	*y = best_y;
	return 1;
}
int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y) {
	//函数实现
	if (NULL == grayImg.data || NULL == subImg.data || grayImg.channels() != 1 || subImg.channels() != 1)
	{
		cout << "image is wrong" << endl;
		return -1;
	}
	if (grayImg.cols < subImg.cols || grayImg.rows < subImg.rows) {
		cout << "size wrong" << endl;
		return -1;
	}
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int sub_h2 = height - sub_height;
	int sub_w2 = width - sub_width;
	float l1, l2;
	float*datay;
	//先计算大图的梯度和幅值
	//梯度的计算
	Mat gradImg_xbig(height, width, CV_32FC1);
	Mat gradImg_ybig(height, width, CV_32FC1);
	Mat magImg_big(height, width, CV_32FC1);
	int u = height*width;
	for (int grad_i1 = 0; grad_i1 <u; grad_i1++)
	{
		float*datax = ((float*)gradImg_xbig.data);
		datax[grad_i1] = 0.0f;
		datay = ((float*)gradImg_ybig.data);
		datay[grad_i1] = 0.0f;
	}
	int grad_rows = height;
	int grad_cols = width;
	for (int grad_i = 1; grad_i < grad_rows - 1; grad_i++) //获得GX,GY
	{
		int grad_need = grad_i*grad_cols;
		for (int grad_j = 1; grad_j < grad_cols - 1; grad_j++) {
			int grad_need1 = grad_need + grad_j;
			l1 = (float)(grayImg.data[grad_need1 + 1 - grad_cols] - grayImg.data[grad_need1 - 1 - grad_cols] + grayImg.data[grad_need1 + 1] + grayImg.data[grad_need1 + 1] - grayImg.data[grad_need1 - 1] - grayImg.data[grad_need1 - 1] + grayImg.data[grad_need1 + 1 + grad_cols] - grayImg.data[grad_need1 - 1 + grad_cols]);
			((float*)gradImg_xbig.data)[grad_need1] = l1;
			l2 = (float)(grayImg.data[grad_need1 - 1 - grad_cols] - grayImg.data[grad_need1 - 1 + grad_cols] + grayImg.data[grad_need1 - grad_cols] + grayImg.data[grad_need1 - grad_cols] - grayImg.data[grad_need1 + grad_cols] - grayImg.data[grad_need1 + grad_cols] + grayImg.data[grad_need1 + 1 - grad_cols] - grayImg.data[grad_need1 + 1 + grad_cols]);
			((float*)gradImg_ybig.data)[grad_need1] = l2;
		}
	}
	//幅值的计算
	int heng = width;
	int shu = height;
	for (int i = 0; i < shu; i++) {
		int need = i*heng;
		for (int j = 0; j < heng; j++) {
			int need1 = i*heng + j;
			((float*)magImg_big.data)[need1] = 0;
		}
	}


	for (int i = 1; i < shu - 1; i++)//计算幅值
	{
		int need = i*heng;
		for (int j = 1; j < heng - 1; j++) {
			int need1 = i*heng + j;
			((float*)magImg_big.data)[need1] = sqrt(((float*)gradImg_xbig.data)[need1] * ((float*)gradImg_xbig.data)[need1] + ((float*)gradImg_ybig.data)[need1] * ((float*)gradImg_ybig.data)[need1]);
		}
	}
	//再计算小图的梯度和幅值
	//梯度的计算
	Mat gradImg_x_small(sub_height, sub_width, CV_32FC1);
	Mat gradImg_y_small(sub_height, sub_width, CV_32FC1);
	Mat magImg_small(sub_height, sub_width, CV_32FC1);
	for (int grad_i = 0; grad_i <sub_height; grad_i++)
	{
		int grad_need0 = grad_i*sub_width;
		for (int grad_j = 0; grad_j <sub_width; grad_j++) {
			int grad_need10 = grad_need0 + grad_j;
			((float*)gradImg_x_small.data)[grad_need10] = 0;
			((float*)gradImg_y_small.data)[grad_need10] = 0;
		}
	}
	for (int grad_i = 1; grad_i < sub_height - 1; grad_i++) //获得GX,GY
	{
		int grad_need = grad_i*sub_width;
		for (int grad_j = 1; grad_j < sub_width - 1; grad_j++) {
			int grad_need1 = grad_need + grad_j;
			float l1_s = (float)(subImg.data[grad_need1 + 1 - sub_width] - subImg.data[grad_need1 - 1 - sub_width] + subImg.data[grad_need1 + 1] + subImg.data[grad_need1 + 1] - subImg.data[grad_need1 - 1] - subImg.data[grad_need1 - 1] + subImg.data[grad_need1 + 1 + sub_width] - subImg.data[grad_need1 - 1 + sub_width]);
			((float*)gradImg_x_small.data)[grad_need1] = l1_s;
			float l2_s = (float)(subImg.data[grad_need1 - 1 - sub_width] - subImg.data[grad_need1 - 1 + sub_width] + subImg.data[grad_need1 - sub_width] + subImg.data[grad_need1 - sub_width] - subImg.data[grad_need1 + sub_width] - subImg.data[grad_need1 + sub_width] + subImg.data[grad_need1 + 1 - sub_width] - subImg.data[grad_need1 + 1 + sub_width]);
			((float*)gradImg_y_small.data)[grad_need1] = l2_s;
		}
	}
	//幅值的计算
	heng = sub_width;
	shu = sub_height;
	for (int i = 0; i < shu; i++) {
		int need = i*heng;
		for (int j = 0; j < heng; j++) {
			int need1 = i*heng + j;
			((float*)magImg_small.data)[need1] = 0;
		}
	}


	for (int i = 1; i < shu - 1; i++)//计算幅值
	{
		int need = i*heng;
		for (int j = 1; j < heng - 1; j++) {
			int need1 = i*heng + j;
			((float*)magImg_small.data)[need1] = sqrt(((float*)gradImg_x_small.data)[need1] * ((float*)gradImg_x_small.data)[need1] + ((float*)gradImg_y_small.data)[need1] * ((float*)gradImg_y_small.data)[need1]);
		}
	}
	//大图小图的幅值已经计算完毕
	//匹配
	Mat searchImg(sub_h2, sub_w2, CV_32FC1);
	int best_x=0, best_y=0;
	int subh_2 = sub_height - 1;
	int subw_2 = sub_width - 1;
	for (int angle_i = 0; angle_i <sub_h2; angle_i++)
	{
		for (int angle_j = 0; angle_j <sub_w2; angle_j++)
		{
			float total_diff = 0;
			//遍历模板图上的每一个像素
			for (int y1 = 1; y1 < subh_2; y1++)
			{
				int row_index = angle_i + y1;
				int love = row_index * width;
				int hate = y1 * sub_width;
				for (int x1 = 1; x1 < subw_2; x1++)
				{
					//大图上的像素位置
					int col_index = angle_j + x1;
					float bigImg_pix = ((float*)magImg_big.data)[love + col_index];
					//模板图上的像素
					float template_pix = ((float*)magImg_small.data)[hate + x1];
					float sub_pix = bigImg_pix - template_pix;
					total_diff += abs(sub_pix);
				}
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[angle_i *sub_w2 + angle_j] = total_diff;
		}
	}
	float minsub = ((float*)searchImg.data)[0];
	for (int i = 0; i < sub_h2; i++)
	{
		for (int j = 0; j < sub_w2; j++)
		{
			float a = ((float*)searchImg.data)[i * sub_w2 + j];
			if (a <minsub) {
				minsub = a;
				best_x = j;
				best_y = i;
			}
		}
	}
	*x = best_x;
	*y = best_y;
	return 1;
}
int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y) {
	//函数实现
	if (NULL == grayImg.data || NULL == subImg.data || grayImg.channels() != 1 || subImg.channels() != 1)
	{
		cout << "image is wrong" << endl;
		return -1;
	}
	if (grayImg.cols < subImg.cols || grayImg.rows < subImg.rows) {
		cout << "size wrong" << endl;
		return -1;
	}
	int* hist_temp = (int *)malloc(256 * sizeof(int));
	int* hist_small = (int *)malloc(256 * sizeof(int));
	ustc_CalcHist(subImg, hist_small, 256);
	for (int i = 0; i < 256; i++) {
		printf("%d\n", hist_small[i]);
	}
	int best_x = 0, best_y = 0;
	int width = grayImg.cols;
	int height = grayImg.rows;
	int sub_width = subImg.cols;
	int sub_height = subImg.rows;
	int sub_h = height - sub_height;
	int sub_w = width - sub_width;

	//该图用于记录每一个像素位置的匹配误差
	Mat searchImg(sub_h, sub_w, CV_32FC1);
	//匹配误差初始化
	//searchImg.setTo(FLT_MAX);

	//遍历大图每一个像素，注意行列的起始、终止坐标
	for (int i = 0; i < 256; i++) {
		hist_temp[i] = 0;
	}
	for (int i = 0; i <sub_h; i++)
	{
		for (int j = 0; j < sub_w; j++)
		{
			//清零
			memset(hist_temp, 0, sizeof(int) * 256);

			//计算当前位置直方图
			for (int x = 0; x < sub_height; x++)
			{
				int col_index = j + x;
				for (int y = 0; y < sub_width; y++)
				{
					//大图上的像素位置
					int row_index = i + y;
					int bigImg_pix = grayImg.data[row_index * width + col_index];
					hist_temp[bigImg_pix]++;
				}
			}

			//根据直方图计算匹配误差
			int total_diff = 0;
			for (int ii = 0; ii < 256; ii++)
			{
				total_diff += abs(hist_temp[ii] - hist_small[ii]);
			}
			//存储当前像素位置的匹配误差
			((float*)searchImg.data)[i * sub_w + j] = total_diff;
		}
	}
	float minsub = ((float*)searchImg.data)[0];
	for (int i = 0; i < sub_h; i++)
	{
		for (int j = 0; j < sub_w; j++)
		{
			float a = ((float*)searchImg.data)[i * sub_w + j];
			if (a < minsub) {
				minsub = a;
				best_x = j;
				best_y = i;
			}
		}
	}
	*x = best_x;
	*y = best_y;
	delete[] hist_temp;
	return 1;
}
