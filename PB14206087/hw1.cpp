//
//  main.cpp
//  opencv3
//
//  Created by chenyan wu on 2017/9/20.
//  Copyright © 2017年 chenyan wu. All rights reserved.
//
/*
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
using namespace std;
using namespace cv;
*/

int ustc_ConvertBgr2Gray(Mat bgrlmg, Mat& graylmg){
    if (NULL == bgrlmg.data)
    {
        cout << "image is null" << endl;
        return -1;
    }
    int width = bgrlmg.cols;
    int height = bgrlmg.rows;
    int i;
    int j;
    int b;
    int g;
    int r;
    int flag=0;
    float grayval;
    for (i=0;i<height;i++)
    {
        for (j=0;j<width;j++)
        {
            b=bgrlmg.data[3*flag+0];
            g=bgrlmg.data[3*flag+1];
            r=bgrlmg.data[3*flag+2];
            grayval=0.114*b+0.587*g+0.229*r;
            graylmg.data[flag]=int(grayval);
            flag++;
        }
    }
    return 1;
}

int ustc_CalcGrad(Mat grayImg, Mat& gradImg_x, Mat& gradImg_y)
{
    if (NULL == grayImg.data)
    {
        cout << "image is NULL." << endl;
        return -1;
    }
    
    int width = grayImg.cols;
    int height = grayImg.rows;
    if ((gradImg_x.cols!=width)||(gradImg_y.cols!=width)||(gradImg_x.rows!=height)||(gradImg_x.rows!=height))
    {
        cout << "The width and the height of gradImag must equal the width and the height of grayImag" << endl;
        return -1;
    }
    
    int i;
    int j;
    int grad_x;
    int grad_y;
    
    for (i = 0; i < height ;i++)
    {
        for (j = 0; j < width ;j++)
        {
            if ((i==0)||(i==height)||(j==0)||(j==width))
            {
                (gradImg_y.data)[i * width + j]=grayImg.data[i* width + j];
                (gradImg_x.data)[i * width + j]=grayImg.data[i* width + j];
            }
            else
            {
                grad_x =
                grayImg.data[(i - 1) * width + j + 1]
                + 2 * grayImg.data[i* width + j + 1]
                + grayImg.data[(i + 1)* width + j + 1]
                - grayImg.data[(i - 1) * width + j - 1]
                - 2 * grayImg.data[i* width + j - 1]
                - grayImg.data[(i + 1)* width + j - 1];
                
                ((float*)gradImg_x.data)[i * width + j] = float(grad_x);
                
                grad_y =
                grayImg.data[(i + 1) * width + j - 1]
                + 2 * grayImg.data[(i + 1)* width + j]
                + grayImg.data[(i + 1)* width + j + 1]
                - grayImg.data[(i - 1) * width + j - 1]
                - 2 * grayImg.data[(i - 1)* width + j ]
                - grayImg.data[(i - 1)* width + j + 1];
                
                ((float*)gradImg_y.data)[i * width + j] = float(grad_y);
                
            }
        }
    }
    return 1;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat& angleImg, Mat& magImg)
{
    if (NULL == gradImg_x.data)
    {
        cout << "gradImg_x is NULL." << endl;
        return -1;
    }
    if (NULL == gradImg_y.data)
    {
        cout << "gradImg_y is NULL." << endl;
        return -1;
    }
    
    int width = gradImg_x.cols;
    int height = gradImg_x.rows;
    float x;
    float y;
    int i;
    int j;
    float val;
    
    for (i = 0; i < height;i++)
    {
        for (j = 0; j < width;j++)
        {
            y=((float*)gradImg_y.data)[i * width + j];
            x=((float*)gradImg_x.data)[i * width + j];
            val=float(atan2(y,x)*180/M_PI);
            if(val<0)
            {
                val+=360.0;
            }
            ((float*)angleImg.data)[i * width + j]=val;
           
            ((float*)magImg.data)[i * width + j]=float(sqrt(x*x+y*y));
            
        }
    }
    return 1;
}
int ustc_Threshold(Mat grayImg, Mat& binaryImg, int th)
{
    if (NULL == grayImg.data)
    {
        cout << "image is NULL." << endl;
        return -1;
    }
    int width = grayImg.cols;
    int height = grayImg.rows;
    int i;
    int j;
    int val;
    
    for (i = 0; i < height;i++)
    {
        for (j = 0; j < width;j++)
        {
            val = grayImg.data[i * width + j];
            if (val>=th)
            {
                binaryImg.data[i * width + j] = 255;
            }
            else
            {
                binaryImg.data[i * width + j] = 0;
            }
            
        }
    }
    return 1;
}
int ustc_CalcHist(Mat grayImg, int* hist, int hist_len)
{
    if (NULL == grayImg.data)
    {
        cout << "image is NULL." << endl;
        return -1;
    }
    int width = grayImg.cols;
    int height = grayImg.rows;
    int i;
    int j;
    int val;
    for (i=0;i<hist_len;i++)
    {
        hist[i]=0;
    }
    
    for (i = 0; i < height;i++)
    {
        for (j = 0; j < width;j++)
        {
            val = grayImg.data[i * width + j];
            hist[val]++;
        }
    }
    return 1;
}
int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int* x, int* y)
{
    //x,y is the left high coodinate of the large image
    if (NULL == grayImg.data)
    {
        cout << "image is NULL." << endl;
        return -1;
    }
    if (NULL == subImg.data)
    {
        cout << "subimage is NULL." << endl;
        return -1;
    }
    int width = grayImg.cols;
    int height = grayImg.rows;
    int subwidth = subImg.cols;
    int subheight = subImg.rows;
    int i;
    int j;
    int u;
    int w;
    int val;
    int min=INT_MAX;
    for (i = 0; i < height-subheight+1;i++)
    {
        for (j = 0; j < width-subwidth+1;j++)
        {
            val=0;
            for (u =0; u < subheight; u++)
            {
                for (w=0; w < subwidth; w++)
                {
                    val+=abs(subImg.data[u*subwidth+w]-grayImg.data[(u+i)*width+j+w]);
                }
                
            }
            if(val<min)
            {
                *x=j;
                *y=i;
                min=val;
            }
            
        }
    }
    return min;
    
}
int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int* x, int* y)
{
    //x,y is the left high coodinate of the large image
    if (NULL == colorImg.data)
    {
        cout << "image is NULL." << endl;
        return -1;
    }
    if (NULL == subImg.data)
    {
        cout << "subimage is NULL." << endl;
        return -1;
    }
    int width = colorImg.cols;
    int height = colorImg.rows;
    int subwidth = subImg.cols;
    int subheight = subImg.rows;
    int i;
    int j;
    int u;
    int w;
    int val;
    int min=INT_MAX;
    for (i = 0; i < height-subheight+1;i++)
    {
        for (j = 0; j < width-subwidth+1;j++)
        {
            val=0;
            for (u =0; u < subheight; u++)
            {
                for (w=0; w < subwidth; w++)
                {
                    val+=abs(subImg.data[3*(u*subwidth+w)]-colorImg.data[3*((u+i)*width+j+w)]);
                    val+=abs(subImg.data[3*(u*subwidth+w)+1]-colorImg.data[3*((u+i)*width+j+w)+1]);
                    val+=abs(subImg.data[3*(u*subwidth+w)+2]-colorImg.data[3*((u+i)*width+j+w)+2]);
                }
                
            }
            if(val<min)
            {
                *x=j;
                *y=i;
                min=val;
            }
            
        }
    }
    return min;
}
int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int* x, int* y)
{
    //x,y is the left high coodinate of the large image
    if (NULL == grayImg.data)
    {
        cout << "image is NULL." << endl;
        return -1;
    }
    if (NULL == subImg.data)
    {
        cout << "subimage is NULL." << endl;
        return -1;
    }
    int width = grayImg.cols;
    int height = grayImg.rows;
    int subwidth = subImg.cols;
    int subheight = subImg.rows;
    int i;
    int j;
    int u;
    int w;
    int val1=0;
    int val2=0;
    int val3=0;
    float max=0;
    float val;
    for (i = 0; i < height-subheight+1;i++)
    {
        for (j = 0; j < width-subwidth+1;j++)
        {
            val=0;
            val1=0;
            val2=0;
            val3=0;
            for (u =0; u < subheight; u++)
            {
                for (w=0; w < subwidth; w++)
                {
                    val1+=subImg.data[u*subwidth+w]*grayImg.data[(u+i)*width+j+w];
                    val2+=subImg.data[u*subwidth+w]*subImg.data[u*subwidth+w];
                    val3+=grayImg.data[(u+i)*width+j+w]*grayImg.data[(u+i)*width+j+w];
                }
                
            }
            val=val1/(sqrt(val2)*sqrt(val3));
            if(val>max)
            {
                *x=j;
                *y=i;
                max=val;
            }
            
        }
        
    }
    return 1;
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int* x, int* y)
{
    if (NULL == grayImg.data)
    {
        cout << "image is NULL." << endl;
        return -1;
    }
    if (NULL == subImg.data)
    {
        cout << "subimage is NULL." << endl;
        return -1;
    }
    int width = grayImg.cols;
    int height = grayImg.rows;
    Mat gradimg_x(height, width, CV_32FC1);
    Mat gradimg_y(height, width, CV_32FC1);
    ustc_CalcGrad(grayImg,gradimg_x,gradimg_y);
    Mat angleimg(height, width, CV_32FC1);
    Mat magimg(height, width, CV_32FC1);
    ustc_CalcAngleMag(gradimg_x,gradimg_y,angleimg,magimg);
    
    
    int subwidth = subImg.cols;
    int subheight = subImg.rows;
    Mat sub_gradimg_x(subheight, subwidth, CV_32FC1);
    Mat sub_gradimg_y(subheight, subwidth, CV_32FC1);
    ustc_CalcGrad(subImg,sub_gradimg_x,sub_gradimg_y);
    Mat sub_angleimg(subheight, subwidth, CV_32FC1);
    Mat sub_magimg(subheight, subwidth, CV_32FC1);
    ustc_CalcAngleMag(sub_gradimg_x,sub_gradimg_y,sub_angleimg,sub_magimg);
    
    
    int i;
    int j;
    int u;
    int w;
    int val;
    int min=INT_MAX;
    for (i = 0; i < height-subheight+1;i++)
    {
        for (j = 0; j < width-subwidth+1;j++)
        {
            val=0;
            for (u =0; u < subheight; u++)
            {
                for (w=0; w < subwidth; w++)
                {
                    val+=abs(int(((float*)sub_angleimg.data)[u*subwidth+w])-int(((float*)angleimg.data)[(u+i)*width+j+w]));
                }
                
            }
            if(val<min)
            {
                *x=j;
                *y=i;
                min=val;
            }
            
        }
    }
    return min;
}


int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int* x, int* y)
{
    if (NULL == grayImg.data)
    {
        cout << "image is NULL." << endl;
        return -1;
    }
    if (NULL == subImg.data)
    {
        cout << "subimage is NULL." << endl;
        return -1;
    }
    int width = grayImg.cols;
    int height = grayImg.rows;
    Mat gradimg_x(height, width, CV_32FC1);
    Mat gradimg_y(height, width, CV_32FC1);
    ustc_CalcGrad(grayImg,gradimg_x,gradimg_y);
    Mat angleimg(height, width, CV_32FC1);
    Mat magimg(height, width, CV_32FC1);
    ustc_CalcAngleMag(gradimg_x,gradimg_y,angleimg,magimg);
    
    
    int subwidth = subImg.cols;
    int subheight = subImg.rows;
    Mat sub_gradimg_x(subheight, subwidth, CV_32FC1);
    Mat sub_gradimg_y(subheight, subwidth, CV_32FC1);
    ustc_CalcGrad(subImg,sub_gradimg_x,sub_gradimg_y);
    Mat sub_angleimg(subheight, subwidth, CV_32FC1);
    Mat sub_magimg(subheight, subwidth, CV_32FC1);
    ustc_CalcAngleMag(sub_gradimg_x,sub_gradimg_y,sub_angleimg,sub_magimg);
    
    
    int i;
    int j;
    int u;
    int w;
    double val;
    float min=FLT_MAX;
    float a;
    float b;
    for (i = 0; i < height-subheight+1;i++)
    {
        for (j = 0; j < width-subwidth+1;j++)
        {
            val=0.0;
            for (u =0; u < subheight; u++)
            {
                for (w=0; w < subwidth; w++)
                {
                    a=((float*)sub_magimg.data)[u*subwidth+w];
                    b=((float*)magimg.data)[(u+i)*width+j+w];
                    val+=abs(a-b)/10000000000000.0;
                }
                
            }
            if(val<min)
            {
                *x=j;
                *y=i;
                min=val;
            }
            //printf("%f\n",val);
        }
    }
    return int(min);
}



int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int* x, int* y)
{
    //x,y is the left high coodinate of the large image
    if (NULL == grayImg.data)
    {
        cout << "image is NULL." << endl;
        return -1;
    }
    if (NULL == subImg.data)
    {
        cout << "subimage is NULL." << endl;
        return -1;
    }
    int width = grayImg.cols;
    int height = grayImg.rows;
    int subwidth = subImg.cols;
    int subheight = subImg.rows;
    int i;
    int j;
    int u;
    int w;
    int val;
    int flag;
    int min=INT_MAX;
    
    
    int sub_list[256];
    int list[256];
    for (i=0; i<256; i++)
    {
        sub_list[i]=0;
    }
    for (u =0; u < subheight; u++)
    {
        for (w=0; w < subwidth; w++)
        {
            flag=subImg.data[u*subwidth+w];
            sub_list[flag]++;
        }
        
    }
    
    for (i = 0; i < height-subheight+1;i++)
    {
        for (j = 0; j < width-subwidth+1;j++)
        {
            for (w=0; w<256; w++)
            {
                list[w]=0;
            }
            val=0;
            for (u =0; u < subheight; u++)
            {
                for (w=0; w < subwidth; w++)
                {
                    flag=grayImg.data[(u+i)*width+j+w];
                    list[flag]++;
                }
                
            }
            for (w=0; w<256; w++)
            {
                val+=abs(list[w]-sub_list[w]);
            }
            if(val<min)
            {
                *x=j;
                *y=i;
                min=val;
            }
            
        }
    }
    
    return 0;
}
/*
int main(int argc, const char * argv[]) {
    Mat image;
    image=imread("/Users/dyh127/Desktop/chenyan/chenyan/opencv3chenyan/opencv3/chenyan.jpg",1);
    //namedWindow("Display Image",WINDOW_AUTOSIZE);
    //imshow("Display Image", image);
    //waitKey(0);
    int width = image.cols;
    int height = image.rows;
    Mat test(height,width,CV_8UC1);
    Mat test2(height,width,CV_8UC1);
    Mat test3(50,50,CV_8UC1);
    Mat test4;
    ustc_ConvertBgr2Gray(image, test);
    Mat imagx(height,width,CV_32FC1);
    Mat imagy(height,width,CV_32FC1);
    ustc_CalcGrad(test, imagx, imagy);
    Mat angleimg(height, width, CV_32FC1);
    Mat magimg(height, width, CV_32FC1);
    ustc_CalcAngleMag(imagx, imagy, angleimg, magimg);
    ustc_Threshold(test, test2, 60);
    int list[256];
    ustc_CalcHist(test, list, 256);
    test3=test(Rect(100,10,50,50)).clone();
    test4=image(Rect(100,10,50,50)).clone();
    int x;
    int y;
    int i;
    int j;
    float val;
    i=ustc_SubImgMatch_hist(test, test3, &x, &y);
    printf("%d %d %d",x,y,i);
    /*
    for (i = 0; i < height;i++)
    {
        for (j = 0; j < width;j++)
        {
            val=((float*)magimg.data)[i*width+j];
            printf("%f ",val);
        }
        printf("\n");
    }
    
     for(i=0;i<256;i++)
     {
     printf("%d ",list[i]);
     }
     */
    namedWindow("grayimg",0);
    imshow("grayimg", test4);
    waitKey(0);
    return 0;
    
    
}
*/
