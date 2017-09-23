#include "SubImageMatch.h"

//函数功能：将bgr图像转化成灰度图像
//bgrImg：彩色图，像素排列顺序是bgr
//grayImg：灰度图，单通道
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg) {
	if (bgrImg.data == NULL || grayImg.data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	if (bgrImg.channels() != 3 || grayImg.channels() != 1)
		return SUB_IMAGE_MATCH_FAIL;
	int bgrRows = bgrImg.rows;
	if (grayImg.rows != bgrRows)
		return SUB_IMAGE_MATCH_FAIL;
	int bgrCols = bgrImg.cols;
	if (grayImg.cols != bgrCols)
		return SUB_IMAGE_MATCH_FAIL;

	if (bgrImg.isContinuous()&&grayImg.isContinuous()) {
		bgrCols *= bgrRows;
		bgrRows = 1;
	}

	int i, j;
	uchar* bgrPtr,* grayPtr;
	for (i = 0; i < bgrRows; ++i) {
		bgrPtr = bgrImg.ptr<uchar>(i);
		grayPtr = grayImg.ptr<uchar>(i);
		for (j = 0; j < bgrCols; ++j) {
			*grayPtr++ = ((*bgrPtr++) * 7472 + (*bgrPtr++) * 38469 + (*bgrPtr++) * 19595) >> 16;
		}
	}
	
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：根据灰度图像计算梯度图像
//grayImg：灰度图，单通道
//gradImg_x：水平方向梯度，浮点类型图像，CV32FC1
//gradImg_y：垂直方向梯度，浮点类型图像，CV32FC1
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y) {
	if (grayImg.data == NULL || gradImg_x.data == NULL || gradImg_y.data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	int imageChannels = grayImg.channels();
	if (imageChannels != 1 || gradImg_x.channels() != imageChannels || gradImg_y.channels() != imageChannels)
		return SUB_IMAGE_MATCH_FAIL;
	int imageRows = grayImg.rows;
	if (gradImg_x.rows != imageRows || gradImg_y.rows != imageRows)
		return SUB_IMAGE_MATCH_FAIL;
	int imageCols = grayImg.cols;
	if (gradImg_x.cols != imageCols || gradImg_y.cols != imageCols)
		return SUB_IMAGE_MATCH_FAIL;
	
	uchar* grayPtr;
	float* gradXPtr, *gradYPtr;
	int i, j;
	float tempL,tempR,tempH,tempV;
	for (i = imageRows - 2; i; i--) {
		grayPtr = grayImg.ptr<uchar>(i);
		grayPtr++;
		gradXPtr = gradImg_x.ptr<float>(i);
		gradXPtr++;
		gradYPtr = gradImg_y.ptr<float>(i);
		gradYPtr++;
		for (j = imageCols - 2; j; j--) {
			tempL = -*(grayPtr - imageCols - 1) + *(grayPtr + imageCols + 1);
			tempR = -*(grayPtr + imageCols - 1) + *(grayPtr - imageCols + 1);
			tempH = -*(grayPtr - 1) + *(grayPtr + 1);
			tempV = -*(grayPtr - imageCols) + *(grayPtr + imageCols);
			grayPtr++;
			*gradXPtr++ = tempL + tempR + 2 * tempH;
			*gradYPtr++ = tempL - tempR + 2 * tempV;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：根据水平和垂直梯度，计算角度和幅值图
//gradImg_x：水平方向梯度，浮点类型图像，CV32FC1
//gradImg_y：垂直方向梯度，浮点类型图像，CV32FC1
//angleImg：角度图，浮点类型图像，CV32FC1
//magImg：幅值图，浮点类型图像，CV32FC1
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg) {
	if (gradImg_x.data == NULL || gradImg_y.data == NULL || angleImg.data == NULL || magImg.data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	int imageChannels = gradImg_x.channels();
	if (imageChannels != 1 || gradImg_y.channels() != imageChannels || angleImg.channels() != imageChannels || magImg.channels() != imageChannels)
		return SUB_IMAGE_MATCH_FAIL;
	int imageRows = gradImg_x.rows;
	if (gradImg_y.rows != imageRows || angleImg.rows != imageRows || magImg.rows != imageRows)
		return SUB_IMAGE_MATCH_FAIL;
	int imageCols = gradImg_x.cols;
	if (gradImg_y.cols != imageCols || angleImg.cols != imageCols || magImg.cols != imageCols)
		return SUB_IMAGE_MATCH_FAIL;

	if (gradImg_x.isContinuous() && gradImg_y.isContinuous() && angleImg.isContinuous() && magImg.isContinuous()) {
		imageCols *= imageRows;
		imageRows = 1;
	}

	int i, j;
	float *gradXPtr, *gradYPtr, *anglePtr, *magPtr;
	for (i = imageRows - 1; i + 1; i--) {
		gradXPtr = gradImg_x.ptr<float>(i);
		gradYPtr = gradImg_y.ptr<float>(i);
		anglePtr = angleImg.ptr<float>(i);
		magPtr = magImg.ptr<float>(i);
		for (j = imageCols; j; j--) {
			*anglePtr = atan2(*gradYPtr, *gradXPtr);
			*anglePtr *= 57.296875;
			if (*anglePtr < 0) *anglePtr += 360;
			*magPtr = sqrt((*gradYPtr)*(*gradYPtr) + (*gradXPtr)*(*gradXPtr));
			gradXPtr++;
			gradYPtr++;
			anglePtr++;
			magPtr++;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：对灰度图像进行二值化
//grayImg：灰度图，单通道
//binaryImg：二值图，单通道
//th：二值化阈值，高于此值，255，低于此值0
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th) {
	if (grayImg.data == NULL || binaryImg.data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	if (grayImg.channels() != 1 || binaryImg.channels() != 1)
		return SUB_IMAGE_MATCH_FAIL;
	int grayRows = grayImg.rows;
	if (binaryImg.rows != grayRows)
		return SUB_IMAGE_MATCH_FAIL;
	int grayCols = grayImg.cols;
	if (binaryImg.cols != grayCols)
		return SUB_IMAGE_MATCH_FAIL;

	if (grayImg.isContinuous() && binaryImg.isContinuous()) {
		grayCols *= grayRows;
		grayRows = 1;
	}

	uchar*grayPtr, *binaryPtr;
	int i, j;
	for (i = 0; i < grayRows; i++) {
		grayPtr = grayImg.ptr<uchar>(i);
		binaryPtr = binaryImg.ptr<uchar>(i);
		for (j = 0; j < grayCols; j++) {
			*binaryPtr++ = ((*grayPtr++ + 256 - th) >> 8) * 255;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：对灰度图像计算直方图
//grayImg：灰度图，单通道
//hist：直方图
//hist_len：直方图的亮度等级，直方图数组的长度
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_CalcHist(Mat grayImg, int* hist, int hist_len) {
	if (grayImg.data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	if (grayImg.channels() != 1)
		return SUB_IMAGE_MATCH_FAIL;
	int imageRows = grayImg.rows;
	int imageCols = grayImg.cols;

	if (grayImg.isContinuous()) {
		imageCols *= imageRows;
		imageRows = 1;
	}

	uchar* grayPtr;
	int i, j;
	memset(hist, 0, 4 * hist_len);
	for (i = imageRows - 1; i + 1; i--) {
		grayPtr = grayImg.ptr<uchar>(i);
		for (j = imageCols; j; j--) {
			hist[*grayPtr]++;
			grayPtr++;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用亮度进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y) {
	if (grayImg.data == NULL || subImg.data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	int imageChannels = grayImg.channels();
	if (imageChannels != 1 || subImg.channels() != imageChannels)
		return SUB_IMAGE_MATCH_FAIL;
	int imageRows = grayImg.rows;
	int subRows = subImg.rows;
	if (subRows > imageRows)
		return SUB_IMAGE_MATCH_FAIL;
	int imageCols = grayImg.cols;
	int subCols = subImg.cols;
	if (subCols > imageCols)
		return SUB_IMAGE_MATCH_FAIL;

	uchar *grayPtr, *subPtr, *keyPtr;
	int i, j, k, l;
	int tempMin = 0x7fffffff;
	int tempSum;
	int location = 0;
	keyPtr = grayImg.data;
	for (i = (imageRows - subRows + 1)*(imageCols - subCols + 1), l = imageCols - subCols; i; l--, i--) {
		grayPtr = keyPtr;
		subPtr = subImg.data;
		tempSum = 0;
		for (j = subRows*subCols, k = subCols - 1; j; j--, k--) {
			tempSum += (!((*grayPtr - *subPtr) >> 31))*(*grayPtr - *subPtr) + (!((*subPtr - *grayPtr) >> 31))*(*subPtr - *grayPtr);
			grayPtr += (!k)*(imageCols - subCols) + 1;
			k += (!k)*subCols;
			subPtr++;
		}
		keyPtr += (!l)*(subCols - 1) + 1;
		l += (!l)*(imageCols - subCols + 1);
		tempMin = !((tempSum - tempMin) >> 31)*tempMin + !((tempMin - tempSum) >> 31)*tempSum;
		location += !(tempSum - tempMin)*(i - location);
	}
	location--;
	location = (imageRows - subRows + 1)*(imageCols - subCols + 1) - location - 1;
	*y = location / (imageCols - subCols + 1);
	*x = location % (imageCols - subCols + 1);
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用色彩进行子图匹配
//colorImg：彩色图，三通单
//subImg：模板子图，三通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y) {
	if (colorImg.data == NULL || subImg.data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	int imageChannels = colorImg.channels();
	if (imageChannels != 3 || subImg.channels() != imageChannels)
		return SUB_IMAGE_MATCH_FAIL;
	int imageRows = colorImg.rows;
	int subRows = subImg.rows;
	if (subRows > imageRows)
		return SUB_IMAGE_MATCH_FAIL;
	int imageCols = colorImg.cols;
	int subCols = subImg.cols;
	if (subCols > imageCols)
		return SUB_IMAGE_MATCH_FAIL;

	uchar *bgrPtr, *subPtr, *keyPtr;
	int i, j, k, l;
	int tempMin = 0x7fffffff;
	int tempSum;
	int location = 0;
	keyPtr = colorImg.data;
	for (i = (imageRows - subRows + 1)*(imageCols - subCols + 1), l = imageCols - subCols; i; l--, i--) {
		bgrPtr = keyPtr;
		subPtr = subImg.data;
		tempSum = 0;
		for (j = subRows*subCols, k = subCols - 1; j; j--, k--) {
			tempSum += (!((*bgrPtr - *subPtr) >> 31))*(*bgrPtr - *subPtr) + (!((*subPtr - *bgrPtr) >> 31))*(*subPtr - *bgrPtr);
			bgrPtr++;
			subPtr++;
			tempSum += (!((*bgrPtr - *subPtr) >> 31))*(*bgrPtr - *subPtr) + (!((*subPtr - *bgrPtr) >> 31))*(*subPtr - *bgrPtr);
			bgrPtr++;
			subPtr++;
			tempSum += (!((*bgrPtr - *subPtr) >> 31))*(*bgrPtr - *subPtr) + (!((*subPtr - *bgrPtr) >> 31))*(*subPtr - *bgrPtr);
			bgrPtr++;
			subPtr++;
			bgrPtr += (!k)*3*(imageCols - subCols);
			k += (!k)*subCols;
		}
		keyPtr += (!l) * 3 * (subCols - 1) + 3;
		l += (!l)*(imageCols - subCols + 1);
		tempMin = !((tempSum - tempMin) >> 31)*tempMin + !((tempMin - tempSum) >> 31)*tempSum;
		location += !(tempSum - tempMin)*(i - location);
	}
	location--;
	location = (imageRows - subRows + 1)*(imageCols - subCols + 1) - location - 1;
	*y = location / (imageCols - subCols + 1);
	*x = location % (imageCols - subCols + 1);
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用亮度相关性进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y) {
	if (grayImg.data == NULL || subImg.data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	int imageChannels = grayImg.channels();
	if (imageChannels != 1 || subImg.channels() != imageChannels)
		return SUB_IMAGE_MATCH_FAIL;
	int imageRows = grayImg.rows;
	int subRows = subImg.rows;
	if (subRows > imageRows)
		return SUB_IMAGE_MATCH_FAIL;
	int imageCols = grayImg.cols;
	int subCols = subImg.cols;
	if (subCols > imageCols)
		return SUB_IMAGE_MATCH_FAIL;

	uchar *grayPtr, *subPtr, *keyPtr;
	int i, j, k, l;
	float tempMax=0;
	long long tempSum = 0;
	long long subSum = 0;
	long long denoSum = 0;
	float flag;
	int loc = 0;
	keyPtr = grayImg.data;
	int* graySqrt = new int[imageRows*imageCols];
	int* p = graySqrt, *keyP = graySqrt;
	for (subPtr = subImg.data, i = subRows*subCols; i; i--) {
		subSum += (*subPtr)*(*subPtr);
		subPtr++;
	}
	for (grayPtr = grayImg.data, i = imageRows * imageCols; i; i--) {
		*p++ = (*grayPtr)*(*grayPtr);
		grayPtr++;
	}
	for (i = (imageRows - subRows + 1)*(imageCols - subCols + 1), l = imageCols - subCols; i; l--, i--) {
		p = keyP;
		grayPtr = keyPtr;
		subPtr = subImg.data;
		tempSum = 0;
		denoSum = 0;
		for (j = subRows*subCols, k = subCols - 1; j; j--, k--) {
			tempSum += (*grayPtr)*(*subPtr);
			denoSum += *p;
			grayPtr += (!k)*(imageCols - subCols) + 1;
			p += (!k)*(imageCols - subCols) + 1;
			k += (!k)*subCols;
			subPtr++;
		}
		keyPtr += (!l)*(subCols - 1) + 1;
		keyP += (!l)*(subCols - 1) + 1;;
		l += (!l)*(imageCols - subCols + 1);
		flag = (float)tempSum*tempSum / (float)(denoSum*subSum);
		if (flag > tempMax) {
			tempMax = flag;
			loc = i;
		}
	}
	loc--;
	loc = (imageRows - subRows + 1)*(imageCols - subCols + 1) - loc - 1;
	*y = loc / (imageCols - subCols + 1);
	*x = loc % (imageCols - subCols + 1);
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用角度值进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y) {
	if (grayImg.data == NULL || subImg.data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	int imageChannels = grayImg.channels();
	if (imageChannels != 1 || subImg.channels() != imageChannels)
		return SUB_IMAGE_MATCH_FAIL;
	int imageRows = grayImg.rows;
	int subRows = subImg.rows;
	if (subRows > imageRows)
		return SUB_IMAGE_MATCH_FAIL;
	int imageCols = grayImg.cols;
	int subCols = subImg.cols;
	if (subCols > imageCols)
		return SUB_IMAGE_MATCH_FAIL;

	uchar* grayPtr, *subPtr;
	int *grayAng, *subAng;
	int *grayP, *subP, *keyP;
	int i, j, k, l;
	int tempL, tempR, tempH, tempV;
	int tempSum;
	int tempMin = 0x7fffffff;
	int location = 0;
	grayAng = new int[(imageRows - 2)*(imageCols - 2)];
	subAng = new int[(subRows - 2)*(subCols - 2)];
	grayP = grayAng;
	subP = subAng;
	keyP = grayAng;

	for (i = 1; i < imageRows - 1; i++) {
		grayPtr = grayImg.ptr<uchar>(i);
		grayPtr++;
		for (j = imageCols - 2; j; j--) {
			tempL = -*(grayPtr - imageCols - 1) + *(grayPtr + imageCols + 1);
			tempR = -*(grayPtr + imageCols - 1) + *(grayPtr - imageCols + 1);
			tempH = -*(grayPtr - 1) + *(grayPtr + 1);
			tempV = -*(grayPtr - imageCols) + *(grayPtr + imageCols);
			grayPtr++;
			*grayP = atan2(tempL - tempR + 2 * tempV, tempL + tempR + 2 * tempH)*57.3;
			*grayP += (*grayP >> 31) & 1 * 360;
			grayP++;
		}
	}
	for (i = 1; i < subRows - 1; i++) {
		subPtr = subImg.ptr<uchar>(i);
		subPtr++;
		for (j = subCols - 2; j; j--) {
			tempL = -*(subPtr - subCols - 1) + *(subPtr + subCols + 1);
			tempR = -*(subPtr + subCols - 1) + *(subPtr - subCols + 1);
			tempH = -*(subPtr - 1) + *(subPtr + 1);
			tempV = -*(subPtr - subCols) + *(subPtr + subCols);
			subPtr++;
			*subP = atan2(tempL - tempR + 2 * tempV, tempL + tempR + 2 * tempH)*57.3;
			*subP += (*subP >> 31) & 1 * 360;
			subP++;
		}
	}
	imageRows -= 2;
	imageCols -= 2;
	subRows -= 2;
	subCols -= 2;
	for (i = (imageRows - subRows + 1)*(imageCols - subCols + 1), l = imageCols - subCols; i; l--, i--) {
		grayP = keyP;
		subP = subAng;
		tempSum = 0;
		for (j = subRows*subCols, k = subCols - 1; j; j--, k--) {
			tempSum += (!((*grayP - *subP) >> 31))*(*grayP - *subP) + (!((*subP - *grayP) >> 31))*(*subP - *grayP);
			grayP += (!k)*(imageCols - subCols) + 1;
			k += (!k)*subCols;
			subP++;
		}
		keyP += (!l)*(subCols - 1) + 1;
		l += (!l)*(imageCols - subCols + 1);
		tempMin = !((tempSum - tempMin) >> 31)*tempMin + !((tempMin - tempSum) >> 31)*tempSum;
		location += !(tempSum - tempMin)*(i - location);
	}
	location--;
	location = (imageRows - subRows + 1)*(imageCols - subCols + 1) - location - 1;
	*y = location / (imageCols - subCols + 1);
	*x = location % (imageCols - subCols + 1);
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用幅值进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y) {
	if (grayImg.data == NULL || subImg.data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	int imageChannels = grayImg.channels();
	if (imageChannels != 1 || subImg.channels() != imageChannels)
		return SUB_IMAGE_MATCH_FAIL;
	int imageRows = grayImg.rows;
	int subRows = subImg.rows;
	if (subRows > imageRows)
		return SUB_IMAGE_MATCH_FAIL;
	int imageCols = grayImg.cols;
	int subCols = subImg.cols;
	if (subCols > imageCols)
		return SUB_IMAGE_MATCH_FAIL;

	uchar* grayPtr, *subPtr;
	int *grayMag, *subMag;
	int *grayP, *subP, *keyP;
	int i, j, k, l;
	int tempL, tempR, tempH, tempV;
	int tempSum;
	int tempMin = 0x7fffffff;
	int location = 0;
	grayMag = new int[(imageRows - 2)*(imageCols - 2)];
	subMag = new int[(subRows - 2)*(subCols - 2)];
	grayP = grayMag;
	subP = subMag;
	keyP = grayMag;
	
	for (i = 1; i < imageRows - 1; i++) {
		grayPtr = grayImg.ptr<uchar>(i);
		grayPtr++;
		for (j = imageCols - 2; j; j--) {
			tempL = -*(grayPtr - imageCols - 1) + *(grayPtr + imageCols + 1);
			tempR = -*(grayPtr + imageCols - 1) + *(grayPtr - imageCols + 1);
			tempH = -*(grayPtr - 1) + *(grayPtr + 1);
			tempV = -*(grayPtr - imageCols) + *(grayPtr + imageCols);
			grayPtr++;
			*grayP = (tempL - tempR + 2 * tempV)*(tempL - tempR + 2 * tempV) + (tempL + tempR + 2 * tempH)*(tempL + tempR + 2 * tempH);
			grayP++;
		}
	}
	for (i = 1; i < subRows - 1; i++) {
		subPtr = subImg.ptr<uchar>(i);
		subPtr++;
		for (j = subCols - 2; j; j--) {
			tempL = -*(subPtr - subCols - 1) + *(subPtr + subCols + 1);
			tempR = -*(subPtr + subCols - 1) + *(subPtr - subCols + 1);
			tempH = -*(subPtr - 1) + *(subPtr + 1);
			tempV = -*(subPtr - subCols) + *(subPtr + subCols);
			subPtr++;
			*subP = (tempL - tempR + 2 * tempV)*(tempL - tempR + 2 * tempV) + (tempL + tempR + 2 * tempH)*(tempL + tempR + 2 * tempH);
			*subP += (*subP >> 31) & 1 * 360;
			subP++;
		}
	}
	imageRows -= 2;
	imageCols -= 2;
	subRows -= 2;
	subCols -= 2;
	for (i = (imageRows - subRows + 1)*(imageCols - subCols + 1), l = imageCols - subCols; i; l--, i--) {
		grayP = keyP;
		subP = subMag;
		tempSum = 0;
		for (j = subRows*subCols, k = subCols - 1; j; j--, k--) {
			tempSum += (!((*grayP - *subP) >> 31))*(*grayP - *subP) + (!((*subP - *grayP) >> 31))*(*subP - *grayP);
			grayP += (!k)*(imageCols - subCols) + 1;
			k += (!k)*subCols;
			subP++;
		}
		keyP += (!l)*(subCols - 1) + 1;
		l += (!l)*(imageCols - subCols + 1);
		tempMin = !((tempSum - tempMin) >> 31)*tempMin + !((tempMin - tempSum) >> 31)*tempSum;
		location += !(tempSum - tempMin)*(i - location);
	}
	location--; tempMin;
	location = (imageRows - subRows + 1)*(imageCols - subCols + 1) - location - 1;
	*y = location / (imageCols - subCols + 1);
	*x = location % (imageCols - subCols + 1);
	return SUB_IMAGE_MATCH_OK;
}

//函数功能：利用直方图进行子图匹配
//grayImg：灰度图，单通道
//subImg：模板子图，单通道
//x：最佳匹配子图左上角x坐标
//y：最佳匹配子图左上角y坐标
//返回值：SUB_IMAGE_MATCH_OK 或者 SUB_IMAGE_MATCH_FAIL

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y) {
	if (grayImg.data == NULL || subImg.data == NULL)
		return SUB_IMAGE_MATCH_FAIL;
	int imageChannels = grayImg.channels();
	if (imageChannels != 1 || subImg.channels() != imageChannels)
		return SUB_IMAGE_MATCH_FAIL;
	int imageRows = grayImg.rows;
	int subRows = subImg.rows;
	if (subRows > imageRows)
		return SUB_IMAGE_MATCH_FAIL;
	int imageCols = grayImg.cols;
	int subCols = subImg.cols;
	if (subCols > imageCols)
		return SUB_IMAGE_MATCH_FAIL;

	int *grayHist = new int[256];
	int *subHist = new int[256];
	Mat matchImg;
	int i, j, k;
	int tempMin = 0x7fffffff;
	int tempSum = 0;
	int l;
	ustc_CalcHist(subImg, subHist, 256);
	for (i = imageRows - subRows; i + 1; i--) {
		for (j = imageCols - subCols; j + 1; j--) {
			matchImg = grayImg(Rect(j, i, subCols, subRows));
			ustc_CalcHist(matchImg, grayHist, 256);
			for (k = 255; k; k--) {
				tempSum += (!((grayHist[k] - subHist[k]) >> 31))*(grayHist[k] - subHist[k]) + (!((subHist[k] - grayHist[k]) >> 31))*(subHist[k] - grayHist[k]);
				if (tempSum != 0)l = k;
			}
			l;
			tempSum += (!((grayHist[0] - subHist[0]) >> 31))*(grayHist[0] - subHist[0]) + (!((subHist[0] - grayHist[0]) >> 31))*(subHist[0] - grayHist[0]);
			if (tempMin > tempSum) {
				tempMin = tempSum;
				*x = j;
				*y = i;
			}
			tempSum = 0;
		}
	}
	return SUB_IMAGE_MATCH_OK;
}
