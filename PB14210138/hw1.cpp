#include "SubImageMatch.h"


#include "stdafx.h"
#include "opencv2/opencv.hpp"
using namespace cv;
#include <iostream>
using namespace std;
#include <time.h>

#define IMG_SHOW
#define MY_OK 1
#define MY_FAIL -1

//#ifndef _H_SUB_IMAGE_MATCH_H_
#define _H_SUB_IMAGE_MATCH_H_

#define SUB_IMAGE_MATCH_OK 1
#define SUB_IMAGE_MATCH_FAIL -1

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat& grayImg)
{
    if (NULL == bgrImg.data )
    {
        cout << "image is NULL." << endl;
        return MY_FAIL;
    }

    int width = bgrImg.cols;
    int height = bgrImg.rows;
    Mat grayImg2(height, width, CV_8UC1);

    for (int row_i = 0; row_i < height; row_i++)
    {
        for (int col_j = 0; col_j < width; col_j += 1)
        {
            int b = bgrImg.data[3 * (row_i * width + col_j) + 0];
            int g = bgrImg.data[3 * (row_i * width + col_j) + 1];
            int r = bgrImg.data[3 * (row_i * width + col_j) + 2];

            int grayVal = b * 114 + g * 587 + r * 229;
            grayVal=grayVal/1000;
            grayImg2.data[row_i * width + col_j] = grayVal;col_j+=1;

            b = bgrImg.data[3 * (row_i * width + col_j) + 0];
            g = bgrImg.data[3 * (row_i * width + col_j) + 1];
            r = bgrImg.data[3 * (row_i * width + col_j) + 2];

            grayVal = b * 114 + g * 587 + r * 229;
            grayVal=grayVal/1000;
            grayImg2.data[row_i * width + col_j] = grayVal;
        }

    }


    grayImg=grayImg2;

}

int ustc_CalcGrad(Mat grayImg,Mat& gradImg_x1, Mat& gradImg_y1)
{
    if (NULL == grayImg.data)
    {
        cout << "image is NULL." << endl;
        return MY_FAIL;
    }

    int width = grayImg.cols;
    int height = grayImg.rows;

    Mat gradImg_x(height, width, CV_32FC1);
    Mat gradImg_y(height, width, CV_32FC1);
    gradImg_x.setTo(0);
    gradImg_y.setTo(0);

    for (int row_i = 1; row_i < height - 1; row_i++)
    {
        for (int col_j = 1; col_j < width - 1; col_j += 1)
        {
            int grad_x =
                    grayImg.data[(row_i - 1) * width + col_j + 1]
                    + 2 * grayImg.data[(row_i)* width + col_j + 1]
                    + grayImg.data[(row_i + 1)* width + col_j + 1]
                    - grayImg.data[(row_i - 1) * width + col_j - 1]
                    - 2 * grayImg.data[(row_i)* width + col_j - 1]
                    - grayImg.data[(row_i + 1)* width + col_j - 1];

            ((float*)gradImg_x.data)[row_i * width + col_j] = grad_x;

            int grad_y=
                   grad_x-2*grayImg.data[(row_i)* width + col_j + 1]
                         +2*grayImg.data[(row_i)* width + col_j - 1]
                         -2*grayImg.data[(row_i-1)* width + col_j]
                         +2*grayImg.data[(row_i+1)* width + col_j];
            ((float*)gradImg_y.data)[row_i * width + col_j] = grad_y;

        }
    }

    gradImg_x1= gradImg_x;
    gradImg_y1=gradImg_y;

}


int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y,Mat& angleImg1, Mat& magImg1)
{
    if (NULL == gradImg_x.data || NULL == gradImg_y.data)
    {
        cout << "image is NULL." << endl;
        return MY_FAIL;
    }

    int width = gradImg_x.cols;
    int height = gradImg_x.rows;

    Mat angleImg(height, width, CV_32FC1);
    Mat magImg(height, width, CV_32FC1);
    angleImg.setTo(0);
    magImg.setTo(0);

    for (int row_i = 1; row_i < height - 1; row_i++)
    {
        for (int col_j = 1; col_j < width - 1; col_j += 1)
        {
            float grad_x = ((float*)gradImg_x.data)[row_i * width + col_j];
            float grad_y = ((float*)gradImg_y.data)[row_i * width + col_j];
            float angle = atan2(grad_y, grad_x);
            float mag1=grad_y*grad_y+grad_x*grad_x;
            int iii;
            float x2, y;
            const float threehalfs = 1.5F;
            x2 = mag * 0.5F;
            y  = mag;
            iii  = * ( int * ) &y;
            iii = 0x5f375a86 - ( i >> 1 );
            y  = * ( float * ) &i;
            y  = y * ( threehalfs - ( x2 * y * y ) );
            y  = y * ( threehalfs - ( x2 * y * y ) );
            y  = y * ( threehalfs - ( x2 * y * y ) );
            mag=y*mag;
            ((float*)angleImg.data)[row_i * width + col_j] = angle;
            ((float*)magImg.data)[row_i * width + col_j] = mag;
        }
    }

    angleImg1= angleImg;
    magImg1=magImg;


}


int ustc_Threshold(Mat grayImg,Mat& binaryImg1, int th)
{
    if (NULL == grayImg.data)
    {
        cout << "image is NULL." << endl;
        return MY_FAIL;
    }

    int width = grayImg.cols;
    int height = grayImg.rows;
    Mat binaryImg(height, width, CV_8UC1);

    int binary_th = th;
    for (int row_i = 0; row_i < height; row_i++)
    {
        int temp0 = row_i * width;
        for (int col_j = 0; col_j < width; col_j += 2)
        {
            int temp1 = temp0 + col_j;
            int pixVal = grayImg.data[temp1];
            int dstVal = 0;
            if (pixVal > binary_th)
            {
                dstVal = 255;
            }
            binaryImg.data[temp1] = dstVal;
        }
    }

    binaryImg1=binaryImg;

    return MY_OK;
}

int ustc_CalcHist(Mat grayImg, int* hist1, int hist_len)
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
        for (int col_j = width ; col_j >=0 ; --col_j)
        {
            int pixVal = grayImg.data[row_i * width + col_j];
            hist[pixVal]++;
        }
    }

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


    Mat searchImg(height, width, CV_32FC1);

    searchImg.setTo(FLT_MAX);

    for (int i = 0; i < height - sub_height; i ++)
    {
        for (int j = 0; j < width - sub_width; j++)
        {
            int total_diff = 0;
            for (int x = 0; x < sub_height; x++)
            {
                for (int y = sub_width; y>=0 ; --y)
                {
                    int row_index = i + y;
                    int col_index = j + x;
                    int bigImg_pix = grayImg.data[row_index * width + col_index];
                    int template_pix = subImg.data[y * sub_width + x];
                    total_diff += abs(bigImg_pix - template_pix);
                }
            }
            ((float*)searchImg.data)[i * width + j] = total_diff;
        }
    }
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y) {
    int colorcol = colorImg.cols, colorrow = colorImg.rows, subcol = subImg.cols, subrow = subImg.rows;
    int stepcol = colorcol - subcol + 1, steprow = colorrow - subrow + 1;
    if (NULL == colorImg.data || NULL == subImg.data || 3 != colorImg.channels() || 3 != subImg.channels() || stepcol <= 0 || steprow <= 0 || NULL == x || NULL == y || CV_8U != colorImg.depth() || CV_8U != subImg.depth()) {   //×ÓÍ¼µÄÆäÖÐÒ»Î¬±ÈÔ­Í¼´óÔò±¨´í
        return SUB_IMAGE_MATCH_FAIL;
    }


    subcol *= 3;
    stepcol *= 3;
    
    int min_total_dif = INT_MAX;
    *x = 0, *y = 0;
    for (int i = 0; i < steprow; i++) {
        for (int j = 0; j < stepcol; j += 3) {
            int total_dif = 0;
            for (int sub_i = 0; sub_i < subrow; sub_i++) {
                uchar *graypixel = colorImg.ptr<uchar>(i + sub_i);
                uchar *subpixel = subImg.ptr<uchar>(sub_i);
                for (int sub_j = 0; sub_j < subcol; sub_j += 3) {
                    int pixel_dif = graypixel[j + sub_j] - subpixel[sub_j];
                    if (pixel_dif > 0) {
                        total_dif += pixel_dif;
                    }
                    else {
                        total_dif += -pixel_dif;
                    }
                    pixel_dif = graypixel[j + sub_j + 1] - subpixel[sub_j + 1];
                    if (pixel_dif > 0) {
                        total_dif += pixel_dif;
                    }
                    else {
                        total_dif += -pixel_dif;
                    }
                    pixel_dif = graypixel[j + sub_j + 2] - subpixel[sub_j + 2];
                    if (pixel_dif > 0) {
                        total_dif += pixel_dif;
                    }
                    else {
                        total_dif += -pixel_dif;
                    }
                }
            }
            if (total_dif < min_total_dif) {
                *x = j / 3, *y = i;
                min_total_dif = total_dif;
            }
        }
    }
    return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y) {
    int graycol = grayImg.cols, grayrow = grayImg.rows, subcol = subImg.cols, subrow = subImg.rows;
    int stepcol = graycol - subcol + 1, steprow = grayrow - subrow + 1;
    if (NULL == grayImg.data || NULL == subImg.data || 1 != grayImg.channels() || 1 != subImg.channels() || stepcol <= 0 || steprow <= 0 || NULL == x || NULL == y || CV_8U != grayImg.depth() || CV_8U != subImg.depth()) {   //×ÓÍ¼µÄÆäÖÐÒ»Î¬±ÈÔ­Í¼´óÔò±¨´í
        return SUB_IMAGE_MATCH_FAIL;
    }
    float max_corr = -1;
    *x = 0, *y = 0;   //×¢Òâ*ºÅ
    for (int i = 0; i < steprow; i++) {
        for (int j = stepcol; j >=0 ; --j) {
            float sum_st = 0;
            float sum_ss = 0;
            float sum_tt = 0;
            for (int sub_i = 0; sub_i < subrow; sub_i++) {
                uchar *graypixel = grayImg.ptr<uchar>(i + sub_i);
                uchar *subpixel = subImg.ptr<uchar>(sub_i);
                for (int sub_j = subcol; sub_j >=0 ; --sub_j) {
                    sum_st += graypixel[j + sub_j] * subpixel[sub_j];
                    sum_ss += graypixel[j + sub_j] * graypixel[j + sub_j];
                    sum_tt += subpixel[sub_j] * subpixel[sub_j];
                }
            }
            float corr = (sum_st *sum_st) / (sum_ss*sum_tt);
            if (corr > max_corr) {
                *x = j, *y = i;
                max_corr = corr;
            }
        }
    }
    return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y) {
    int graycol = grayImg.cols, grayrow = grayImg.rows, subcol = subImg.cols, subrow = subImg.rows;
    int stepcol = graycol - subcol + 1, steprow = grayrow - subrow + 1;
    if (stepcol <= 0 || steprow <= 0 || NULL == x || NULL == y) {   //ÆäËûÅÐ¶ÏºóÃæº¯Êý¶¼ÓÐ×ö
        return SUB_IMAGE_MATCH_FAIL;
    }
    Mat gradImg_x, gradImg_y, subgradImg_x, subgradImg_y;
    if (SUB_IMAGE_MATCH_FAIL == ustc_CalcGrad(grayImg, gradImg_x, gradImg_y) || SUB_IMAGE_MATCH_FAIL == ustc_CalcGrad(subImg, subgradImg_x, subgradImg_y)) {
        return SUB_IMAGE_MATCH_FAIL;
    }
    Mat angleImg, magImg, subangleImg, submagImg;
    if (SUB_IMAGE_MATCH_FAIL == ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg) || SUB_IMAGE_MATCH_FAIL == ustc_CalcAngleMag(subgradImg_x, subgradImg_y, subangleImg, submagImg)) {
        return SUB_IMAGE_MATCH_FAIL;
    }
    int min_total_dif = INT_MAX;
    *x = 0, *y = 0;   //×¢Òâ*ºÅ
    for (int i = 0; i < steprow; i++) {
        for (int j = 0; j < stepcol; j++) {
            int total_dif = 0;
            for (int sub_i = subrow; sub_i >=0 ; --sub_i) {
                float *anglepixel = angleImg.ptr<float>(i + sub_i);
                float *subanglepixel = subangleImg.ptr<float>(sub_i);
                for (int sub_j = subcol; sub_j >=0 ; --sub_j) {
                    int pixel_dif = (int)(anglepixel[j + sub_j]) - (int)(subanglepixel[sub_j]);  //¿ìËÙ×ª»¯ÎªÕûÊý£¿
                    if (pixel_dif < 0) {
                        pixel_dif = -pixel_dif;
                    }
                    if (pixel_dif < 180) {
                        total_dif += pixel_dif;
                    }
                    else {
                        total_dif += 360 - pixel_dif;
                    }
                }
            }
            if (total_dif < min_total_dif) {
                *x = j, *y = i;
                min_total_dif = total_dif;
            }
        }
    }
    return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y) {
    int graycol = grayImg.cols, grayrow = grayImg.rows, subcol = subImg.cols, subrow = subImg.rows;
    int stepcol = graycol - subcol + 1, steprow = grayrow - subrow + 1;
    if (stepcol <= 0 || steprow <= 0 || NULL == x || NULL == y) {   //ÆäËûÅÐ¶ÏºóÃæº¯Êý¶¼ÓÐ×ö
        return SUB_IMAGE_MATCH_FAIL;
    }
    Mat gradImg_x, gradImg_y, subgradImg_x, subgradImg_y;
    if (SUB_IMAGE_MATCH_FAIL == ustc_CalcGrad(grayImg, gradImg_x, gradImg_y) || SUB_IMAGE_MATCH_FAIL == ustc_CalcGrad(subImg, subgradImg_x, subgradImg_y)) {
        return SUB_IMAGE_MATCH_FAIL;
    }
    Mat angleImg, magImg, subangleImg, submagImg;
    if (SUB_IMAGE_MATCH_FAIL == ustc_CalcAngleMag(gradImg_x, gradImg_y, angleImg, magImg) || SUB_IMAGE_MATCH_FAIL == ustc_CalcAngleMag(subgradImg_x, subgradImg_y, subangleImg, submagImg)) {
        return SUB_IMAGE_MATCH_FAIL;
    }
    int min_total_dif = INT_MAX;
    *x = 0, *y = 0;   //×¢Òâ*ºÅ
    for (int i = 0; i < steprow; i++) {
        for (int j = 0; j < stepcol; j++) {
            int total_dif = 0;
            for (int sub_i = subrow; sub_i >=0 ; --sub_i) {
                float *magpixel = magImg.ptr<float>(i + sub_i);
                float *submagpixel = submagImg.ptr<float>(sub_i);
                for (int sub_j = subcol; sub_j >=0 ; --sub_j) {
                    int pixel_dif = (int)(magpixel[j + sub_j]) - (int)(submagpixel[sub_j]);
                    if (pixel_dif > 0) {
                        total_dif += pixel_dif;
                    }
                    else {
                        total_dif += -pixel_dif;
                    }
                    //total_dif += (pixel_dif > 0) ? pixel_dif : -pixel_dif;
                }
            }
            if (total_dif < min_total_dif) {
                *x = j, *y = i;
                min_total_dif = total_dif;
            }
        }
    }
    return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y) {
    int graycol = grayImg.cols, grayrow = grayImg.rows, subcol = subImg.cols, subrow = subImg.rows;
    int stepcol = graycol - subcol + 1, steprow = grayrow - subrow + 1;
    if (NULL == grayImg.data || 1 != grayImg.channels() || stepcol <= 0 || steprow <= 0 || NULL == x || NULL == y || CV_8U != grayImg.depth()) {   //subÒÑÔÚÏÂÃæÅÐ¶Ï
        return SUB_IMAGE_MATCH_FAIL;
    }
    int* hist_temp = new int[256];
    int* sub_hist_temp = new int[256];
    if (SUB_IMAGE_MATCH_FAIL == ustc_CalcHist(subImg, sub_hist_temp, 256)) {
        return SUB_IMAGE_MATCH_FAIL;
    }
    int min_total_dif = INT_MAX;
    *x = 0, *y = 0;   //×¢Òâ*ºÅ
    for (int i = 0; i < steprow; i++) {
        for (int j = 0; j < stepcol; j++) {
            for (int k = 256; k >=0 ; --k) {
                hist_temp[k] = 0;
            }
            for (int sub_i = 0; sub_i < subrow; sub_i++) {
                uchar *graypixel = grayImg.ptr<uchar>(i + sub_i);
                for (int sub_j = subcol; sub_j >=0 ; --sub_j) {
                    hist_temp[graypixel[j + sub_j]]++;
                }
            }
            int total_dif = 0;
            for (int k = 256; k >=0 ; --k) {
                int hist_dif = hist_temp[k] - sub_hist_temp[k];
                if (hist_dif > 0) {
                    total_dif += hist_dif;
                }
                else {
                    total_dif += -hist_dif;
                }
                //total_dif += (hist_dif > 0) ? hist_dif : -hist_dif;
            }
            if (total_dif < min_total_dif) {
                *x = j, *y = i;
                min_total_dif = total_dif;
            }
        }
    }
    delete[] hist_temp;
    delete[] sub_hist_temp;
    return SUB_IMAGE_MATCH_OK;
}
