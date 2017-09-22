//  SubImageMatch.cpp
//  SubImageMatch
//
//  Created by Lixian on 2017/9/11.
//  Copyright © 2017年 Lixian. All rights reserved.
//

/* Includes ------------------------------------------------------------------*/
#include "SubImageMatch.h"
#include <xmmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>// SSE2:<e*.h>, SSE3:<p*.h>, SSE4:<s*.h>
/* Private define ------------------------------------------------------------*/
#define USE_SSE2
/* Private macro -------------------------------------------------------------*/
#ifdef _MSC_VER /* visual c++ */
#define ALIGN16_BEG __declspec(align(16))
#define ALIGN16_END
#else /* gcc or icc */
#define ALIGN16_BEG
#define ALIGN16_END __attribute__((aligned(16)))
#endif

#define _PS_CONST(Name, Val)                                            \
static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PI32_CONST(Name, Val)                                            \
static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PS_CONST_TYPE(Name, Type, Val)                                 \
static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
/* Private constant ----------------------------------------------------------*/
_PS_CONST(0, 0);
_PS_CONST(1, 1.0f);

/* the smallest non denormalized float number */
_PS_CONST_TYPE(sign_mask, int, (int) 0x80000000);
_PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PS_CONST(atanrange_hi, 2.414213562373095);
_PS_CONST(atanrange_lo, 0.4142135623730950);
_PS_CONST(cephes_2PIF, 6.28318530717958648);
_PS_CONST(cephes_PIF, 3.141592653589793238);
_PS_CONST(cephes_PIO2F, 1.5707963267948966192);
_PS_CONST(cephes_PIO4F, 0.7853981633974483096);

_PS_CONST(atancof_p0, 8.05374449538e-2);
_PS_CONST(atancof_p1, 1.38776856032E-1);
_PS_CONST(atancof_p2, 1.99777106478E-1);
_PS_CONST(atancof_p3, 3.33329491539E-1);

_PS_CONST(pitodeg, 57.29577951);

_PS_CONST(minuspio2, -1.5707963267948966192);

__m128i ZERO = _mm_setzero_si128();
/* Private subfunctions ------------------------------------------------------*/
/* for convenience of calling simd sqrt */
float sqrt_ps(float x) {
    __m128 sse_value = _mm_set_ps1(x);
    sse_value = _mm_sqrt_ps(sse_value);
    return _mm_cvtss_f32(sse_value);
}

float rsqrt_ps(float x) {
    __m128 sse_value = _mm_set_ps1(x);
    sse_value = _mm_rsqrt_ps(sse_value);
    return _mm_cvtss_f32(sse_value);
}


__m128 atan_ps(__m128 x) {
    __m128 sign_bit, y;
    
    sign_bit = x;
    /* take the absolute value */
    x = _mm_and_ps(x, *(__m128 *) _ps_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit = _mm_and_ps(sign_bit, *(__m128 *) _ps_sign_mask);
    
    /* range reduction, init x and y depending on range */
#ifdef USE_SSE2
    /* x > 2.414213562373095 */
    __m128 cmp0 = _mm_cmpgt_ps(x, *(__m128 *) _ps_atanrange_hi);
    /* x > 0.4142135623730950 */
    __m128 cmp1 = _mm_cmpgt_ps(x, *(__m128 *) _ps_atanrange_lo);
    
    /* x > 0.4142135623730950 && !( x > 2.414213562373095 ) */
    __m128 cmp2 = _mm_andnot_ps(cmp0, cmp1);
    
    /* -( 1.0/x ) */
    __m128 y0 = _mm_and_ps(cmp0, *(__m128 *) _ps_cephes_PIO2F);
    __m128 x0 = _mm_div_ps(*(__m128 *) _ps_1, x);
    x0 = _mm_xor_ps(x0, *(__m128 *) _ps_sign_mask);
    
    __m128 y1 = _mm_and_ps(cmp2, *(__m128 *) _ps_cephes_PIO4F);
    /* (x-1.0)/(x+1.0) */
    __m128 x1_o = _mm_sub_ps(x, *(__m128 *) _ps_1);
    __m128 x1_u = _mm_add_ps(x, *(__m128 *) _ps_1);
    __m128 x1 = _mm_div_ps(x1_o, x1_u);
    
    __m128 x2 = _mm_and_ps(cmp2, x1);
    x0 = _mm_and_ps(cmp0, x0);
    x2 = _mm_or_ps(x2, x0);
    cmp1 = _mm_or_ps(cmp0, cmp2);
    x2 = _mm_and_ps(cmp1, x2);
    x = _mm_andnot_ps(cmp1, x);
    x = _mm_or_ps(x2, x);
    
    y = _mm_or_ps(y0, y1);
#else
#error sse1 & mmx version not implemented
#endif
    
    __m128 zz = _mm_mul_ps(x, x);
    __m128 acc = *(__m128 *) _ps_atancof_p0;
    acc = _mm_mul_ps(acc, zz);
    acc = _mm_sub_ps(acc, *(__m128 *) _ps_atancof_p1);
    acc = _mm_mul_ps(acc, zz);
    acc = _mm_add_ps(acc, *(__m128 *) _ps_atancof_p2);
    acc = _mm_mul_ps(acc, zz);
    acc = _mm_sub_ps(acc, *(__m128 *) _ps_atancof_p3);
    acc = _mm_mul_ps(acc, zz);
    acc = _mm_mul_ps(acc, x);
    acc = _mm_add_ps(acc, x);
    y = _mm_add_ps(y, acc);
    
    /* update the sign */
    y = _mm_xor_ps(y, sign_bit);
    
    return y;
}

__m128 atan2_ps(__m128 y, __m128 x) {
    __m128 x_eq_0 = _mm_cmpeq_ps(x, *(__m128 *) _ps_0);
    __m128 x_gt_0 = _mm_cmpgt_ps(x, *(__m128 *) _ps_0);
    __m128 x_le_0 = _mm_cmple_ps(x, *(__m128 *) _ps_0);
    __m128 y_eq_0 = _mm_cmpeq_ps(y, *(__m128 *) _ps_0);
    __m128 x_lt_0 = _mm_cmplt_ps(x, *(__m128 *) _ps_0);
    __m128 y_lt_0 = _mm_cmplt_ps(y, *(__m128 *) _ps_0);
    
    __m128 zero_mask = _mm_and_ps(x_eq_0, y_eq_0);
    __m128 zero_mask_other_case = _mm_and_ps(y_eq_0, x_gt_0);
    zero_mask = _mm_or_ps(zero_mask, zero_mask_other_case);
    
    __m128 pio2_mask = _mm_andnot_ps(y_eq_0, x_eq_0);
    __m128 pio2_mask_sign = _mm_and_ps(y_lt_0, *(__m128 *) _ps_sign_mask);
    __m128 pio2_result = *(__m128 *) _ps_cephes_PIO2F;
    pio2_result = _mm_xor_ps(pio2_result, pio2_mask_sign);
    pio2_result = _mm_and_ps(pio2_mask, pio2_result);
    
    __m128 pi_mask = _mm_and_ps(y_eq_0, x_le_0);
    __m128 pi = *(__m128 *) _ps_cephes_PIF;
    __m128 pi_result = _mm_and_ps(pi_mask, pi);
    
    __m128 swap_sign_mask_offset = _mm_and_ps(x_lt_0, y_lt_0);
    swap_sign_mask_offset = _mm_and_ps(swap_sign_mask_offset, *(__m128 *) _ps_sign_mask);
    
    __m128 offset0 = _mm_setzero_ps();
    __m128 offset1 = *(__m128 *) _ps_cephes_PIF;
    offset1 = _mm_xor_ps(offset1, swap_sign_mask_offset);
    
    __m128 offset = _mm_andnot_ps(x_lt_0, offset0);
    offset = _mm_and_ps(x_lt_0, offset1);
    
    __m128 arg = _mm_div_ps(y, x);
    __m128 atan_result = atan_ps(arg);
    atan_result = _mm_add_ps(atan_result, offset);
    
    /* select between zero_result, pio2_result and atan_result */
    
    __m128 result = _mm_andnot_ps(zero_mask, pio2_result);
    atan_result = _mm_andnot_ps(pio2_mask, atan_result);
    atan_result = _mm_andnot_ps(pio2_mask, atan_result);
    result = _mm_or_ps(result, atan_result);
    result = _mm_or_ps(result, pi_result);
    result = _mm_max_ps(result, *(__m128 *) _ps_minuspio2);
    
    return result;
}

static inline __m128i _mm_cmpgt_epu8(const __m128i x, const __m128i y) {
    // Returns 0xFF where x > y:
    return _mm_andnot_si128(
                            _mm_cmpeq_epi8(x, y),
                            _mm_cmpeq_epi8(_mm_max_epu8(x, y), x)
                            );
}// Compare each of the 8-bit unsigned ints in x to those in y and set the result to 0xFF where x > y. Equivalent to checking whether x is equal to the maximum of x and y, but not equal to y itself.


/* Private functions ---------------------------------------------------------*/

int ustc_ConvertBgr2Gray(Mat bgrImg, Mat &grayImg) {
    
    if (NULL == bgrImg.data) {
        return SUB_IMAGE_MATCH_FAIL;
    }
    
    int sizeOfPixel = bgrImg.cols * bgrImg.rows;        //像素数
    uchar *gray = (uchar *) grayImg.data;
    uchar *bgr = bgrImg.data;
    
    __m128i A1, A2, A3, B1, B2, G1, G2, R1, R2, C1, C2, D;
    for (int i = 0; i < sizeOfPixel; i += 16) {
        A1 = _mm_loadu_si128((__m128i *) (bgr + 3 * i));
        A2 = _mm_loadu_si128((__m128i *) (bgr + 3 * i + 16));
        A3 = _mm_loadu_si128((__m128i *) (bgr + 3 * i + 32));
        
        B1 = _mm_shuffle_epi8(A1, _mm_setr_epi8(0, -1, 3, -1, 6, -1, 9, -1, 12, -1, 15, -1, -1, -1, -1, -1));
        B1 = _mm_or_si128(B1, _mm_shuffle_epi8(A2,
                                               _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, 5,
                                                             -1)));
        B2 = _mm_shuffle_epi8(A2, _mm_setr_epi8(8, -1, 11, -1, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
        B2 = _mm_or_si128(B2, _mm_shuffle_epi8(A3,
                                               _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 1, -1, 4, -1, 7, -1, 10, -1, 13,
                                                             -1)));
        
        G1 = _mm_shuffle_epi8(A1, _mm_setr_epi8(1, -1, 4, -1, 7, -1, 10, -1, 13, -1, -1, -1, -1, -1, -1, -1));
        G1 = _mm_or_si128(G1, _mm_shuffle_epi8(A2,
                                               _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 3, -1, 6,
                                                             -1)));
        G2 = _mm_shuffle_epi8(A2, _mm_setr_epi8(9, -1, 12, -1, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
        G2 = _mm_or_si128(G2, _mm_shuffle_epi8(A3,
                                               _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, -1, 5, -1, 8, -1, 11, -1, 14,
                                                             -1)));
        
        R1 = _mm_shuffle_epi8(A1, _mm_setr_epi8(2, -1, 5, -1, 8, -1, 11, -1, 14, -1, -1, -1, -1, -1, -1, -1));
        R1 = _mm_or_si128(R1, _mm_shuffle_epi8(A2,
                                               _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 4, -1, 7,
                                                             -1)));
        R2 = _mm_shuffle_epi8(A2, _mm_setr_epi8(10, -1, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
        R2 = _mm_or_si128(R2, _mm_shuffle_epi8(A3, _mm_setr_epi8(-1, -1, -1, -1, 0, -1, 3, -1, 6, -1, 9, -1, 12, -1, 15,
                                                                 -1)));
        
        C1 = _mm_srli_epi16(_mm_add_epi16(
                                          _mm_add_epi16(_mm_mullo_epi16(B1, _mm_set1_epi16(15)), _mm_mullo_epi16(G1, _mm_set1_epi16(75))),
                                          _mm_mullo_epi16(R1, _mm_set1_epi16(38))), 7);
        C2 = _mm_srli_epi16(_mm_add_epi16(
                                          _mm_add_epi16(_mm_mullo_epi16(B2, _mm_set1_epi16(15)), _mm_mullo_epi16(G2, _mm_set1_epi16(75))),
                                          _mm_mullo_epi16(R2, _mm_set1_epi16(38))), 7);
        
        D = _mm_packus_epi16(C1, C2);
        _mm_store_si128((__m128i *) (gray + i), D);
        
    }
    
    return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcGrad(Mat grayImg, Mat &gradImg_x, Mat &gradImg_y) {
    
    if (NULL == grayImg.data) {
        return SUB_IMAGE_MATCH_FAIL;
    }
    
    uchar *P = grayImg.data;
    float *PX = (float *) gradImg_x.data;
    float *PY = (float *) gradImg_y.data;
    
    int rows = grayImg.rows;
    int cols = grayImg.cols;
    
    //    gradImg_x.setTo(0);
    //    gradImg_y.setTo(0);
    for (int i = 1; i < rows - 1; i++) {
        int j = 1;
        for (; j < cols - 1 - 4; j += 4) {
            //通过指针遍历图像上每一个像素
            
            PX[i * cols + j] = P[(i - 1) * cols + j + 1] + (P[i * cols + j + 1] << 1) + P[(i + 1) * cols + j + 1] -
            P[(i - 1) * cols + j - 1] - (P[i * cols + j - 1] << 1) - P[(i + 1) * cols + j - 1];
            PY[i * cols + j] = P[(i + 1) * cols + j - 1] + (P[(i + 1) * cols + j] << 1) + P[(i + 1) * cols + j + 1] -
            P[(i - 1) * cols + j - 1] - (P[(i - 1) * cols + j] << 1) - P[(i - 1) * cols + j + 1];
            
            PX[i * cols + j + 1] = P[(i - 1) * cols + j + 2] + (P[i * cols + j + 2] << 1) + P[(i + 1) * cols + j + 2] -
            P[(i - 1) * cols + j] - (P[i * cols + j] << 1) - P[(i + 1) * cols + j];
            PY[i * cols + j + 1] =
            P[(i + 1) * cols + j] + (P[(i + 1) * cols + j + 1] << 1) + P[(i + 1) * cols + j + 2] -
            P[(i - 1) * cols + j] - (P[(i - 1) * cols + j + 1] << 1) - P[(i - 1) * cols + j + 2];
            
            PX[i * cols + j + 2] = P[(i - 1) * cols + j + 3] + (P[i * cols + j + 3] << 1) + P[(i + 1) * cols + j + 3] -
            P[(i - 1) * cols + j + 1] - (P[i * cols + j + 1] << 1) - P[(i + 1) * cols + j + 1];
            PY[i * cols + j + 2] =
            P[(i + 1) * cols + j + 1] + (P[(i + 1) * cols + j + 2] << 1) + P[(i + 1) * cols + j + 3] -
            P[(i - 1) * cols + j + 1] - (P[(i - 1) * cols + j + 2] << 1) - P[(i - 1) * cols + j + 3];
            
            PX[i * cols + j + 3] = P[(i - 1) * cols + j + 4] + (P[i * cols + j + 4] << 1) + P[(i + 1) * cols + j + 4] -
            P[(i - 1) * cols + j + 2] - (P[i * cols + j + 2] << 1) - P[(i + 1) * cols + j + 2];
            PY[i * cols + j + 3] =
            P[(i + 1) * cols + j + 2] + (P[(i + 1) * cols + j + 3] << 1) + P[(i + 1) * cols + j + 4] -
            P[(i - 1) * cols + j + 2] - (P[(i - 1) * cols + j + 3] << 1) - P[(i - 1) * cols + j + 4];
        }
        for (; j < cols - 1; j++) {
            PX[i * cols + j] = P[(i - 1) * cols + j + 1] + (P[i * cols + j + 1] << 1) + P[(i + 1) * cols + j + 1] -
            P[(i - 1) * cols + j - 1] - (P[i * cols + j - 1] << 1) - P[(i + 1) * cols + j - 1];
            PY[i * cols + j] = P[(i + 1) * cols + j - 1] + (P[(i + 1) * cols + j] << 1) + P[(i + 1) * cols + j + 1] -
            P[(i - 1) * cols + j - 1] - (P[(i - 1) * cols + j] << 1) - P[(i - 1) * cols + j + 1];
        }
    }
    
    return SUB_IMAGE_MATCH_OK;
}

int ustc_CalcAngleMag(Mat gradImg_x, Mat gradImg_y, Mat &angleImg, Mat &magImg) {
    
    if (NULL == gradImg_y.data || NULL == gradImg_x.data) {
        return SUB_IMAGE_MATCH_FAIL;
    }
    
    int sizeOfPixel = gradImg_x.cols * gradImg_x.rows;        //像素数
    
    float *PX = (float *) gradImg_x.data;
    float *PY = (float *) gradImg_y.data;
    float *PA = (float *) angleImg.data;
    float *PM = (float *) magImg.data;
    __m128 A1, B1, C1, D1, E1, F1;
    
    int i = 0;
    
    for (; i < sizeOfPixel - 4; i += 4) {
        A1 = _mm_load_ps(PX + i);
        B1 = _mm_load_ps(PY + i);
        C1 = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(A1, A1), _mm_mul_ps(B1, B1)));
        _mm_store_ps(PM + i, C1);
        D1 = atan2_ps(B1, A1);
        E1 = _mm_cmplt_ps(D1, *(__m128 *) _ps_0);
        E1 = _mm_and_ps(E1, *(__m128 *) _ps_1);
        F1 = _mm_mul_ps(_mm_add_ps(D1, _mm_mul_ps(E1, *(__m128 *) _ps_cephes_2PIF)), *(__m128 *) _ps_pitodeg);
        _mm_store_ps(PA + i, F1);
    }
    for (; i < sizeOfPixel; i++) {
        PM[i] = sqrt(PX[i] * PX[i] + PY[i] + PY[i]);
        PA[i] = atan2(PY[i], PX[i]);
        if (PA[i] < 0) PA[i] = PA[i] * 57.29577951 + 360;
        else PA[i] = PA[i] * 57.2957795;
    }
    
    return SUB_IMAGE_MATCH_OK;
};

int ustc_Threshold(Mat grayImg, Mat &binaryImg, int th) {
    
    if (NULL == grayImg.data) {
        return SUB_IMAGE_MATCH_FAIL;
    }
    if (th <= 0) {
        binaryImg.setTo(255);
        return SUB_IMAGE_MATCH_OK;
    } else if (th >= 255) {
        binaryImg.setTo(0);
        return SUB_IMAGE_MATCH_OK;
    }
    
    int sizeOfPixel = grayImg.cols * grayImg.rows;
    int qsizeOfPixel = sizeOfPixel >> 2;        //像素数/4
    int fqsizeOfPixel = qsizeOfPixel << 2;
    
    uint32_t *pg = (uint32_t *) grayImg.data;
    uint32_t *pb = (uint32_t *) binaryImg.data;
    __m128i A, B;
    
    uchar tth = (uchar) th;
    uchar pth[16] = {tth, tth, tth, tth, tth, tth, tth, tth, tth, tth, tth, tth, tth, tth, tth, tth};
    B = _mm_load_si128((__m128i *) ((uint32_t *) pth));
    
    for (int i = 0; i < qsizeOfPixel; i += 4) {
        A = _mm_load_si128((__m128i *) (pg + i));
        A = _mm_cmpgt_epu8(A, B);
        _mm_store_si128((__m128i *) (pb + i), A);
    }
    for (int i = fqsizeOfPixel; i < sizeOfPixel; i++) {
        binaryImg.data[i] = (((grayImg.data[i] - th) >> 31) & 0x1) - 1;
    }
    
    return SUB_IMAGE_MATCH_OK;
}


int ustc_CalcHist(Mat grayImg, int *hist, int hist_len) {
    
    if (NULL == grayImg.data || hist_len != 256) {
        return SUB_IMAGE_MATCH_FAIL;
    }
    
    int sizeOfPixel = grayImg.cols * grayImg.rows;
    //    int qsizeOfPixel = sizeOfPixel >> 2;        //像素数/4
    //    int fqsizeOfPixel = qsizeOfPixel << 2;
    
    uchar *pg = grayImg.data;
    
    
    int *hist1 = new int[hist_len]();
    int *hist2 = new int[hist_len]();
    int *hist3 = new int[hist_len]();
    int *hist4 = new int[hist_len]();
    
    //    for (int i = 0; i < hist_len; i++) hist[i] = 0;
    int i = 0;
    
    //Parallel
    for (; i < sizeOfPixel - 16; i += 16) {
        hist1[pg[i]] += 1;
        hist2[pg[i + 1]] += 1;
        hist3[pg[i + 2]] += 1;
        hist4[pg[i + 3]] += 1;
        
        hist1[pg[i + 4]] += 1;
        hist2[pg[i + 5]] += 1;
        hist3[pg[i + 6]] += 1;
        hist4[pg[i + 7]] += 1;
        
        hist1[pg[i + 8]] += 1;
        hist2[pg[i + 9]] += 1;
        hist3[pg[i + 10]] += 1;
        hist4[pg[i + 11]] += 1;
        
        hist1[pg[i + 12]] += 1;
        hist2[pg[i + 13]] += 1;
        hist3[pg[i + 14]] += 1;
        hist4[pg[i + 15]] += 1;
        
    }
    for (; i < sizeOfPixel; i++) {
        hist[pg[i]] += 1;
    }
    for (int i = 0; i < hist_len; i++) {
        hist[i] = hist1[i] + hist2[i] + hist3[i] + hist4[i];
    }
    delete[]hist1;
    delete[]hist2;
    delete[]hist3;
    delete[]hist4;
    return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_gray(Mat grayImg, Mat subImg, int *x, int *y) {
    
    if (NULL == grayImg.data || NULL == subImg.data || grayImg.rows < subImg.rows || grayImg.cols < subImg.cols) {
        return SUB_IMAGE_MATCH_FAIL;
    }
    
    uchar *pg = grayImg.data;
    uchar *ps = subImg.data;
    const int rows = grayImg.rows;
    const int cols = grayImg.cols;
    const int subrows = subImg.rows;
    const int subcols = subImg.cols;
    const int diffrows = rows - subrows;
    const int diffcols = cols - subcols;
    int i, j, k, l;
    
    int sum = 0;
    int littlesum = 0x4fffffff;
    (*x) = 0;
    (*y) = 0;
    
    __m128i SAD2, SUM_ALL;
    
    for (i = 0; i <= diffrows; i++) {
        for (j = 0; j <= diffcols; j++) {
            sum = 0;
            SUM_ALL = ZERO;
            for (k = 0; k < subrows; k++) {
                for (l = 0; l < subcols - 16; l += 16) {
                    SAD2 = _mm_sad_epu8(_mm_loadu_si128((__m128i *) (pg + (i + k) * cols + j + l)),
                                        _mm_loadu_si128((__m128i *) (ps + k * subcols + l)));
                    SUM_ALL = _mm_add_epi64(SUM_ALL, SAD2);
                }
                for (; l < subcols; l++) {
                    sum += abs((int) pg[(i + k) * cols + j + l] - ps[k * subcols + l]);
                }
            }
            
            SUM_ALL = _mm_add_epi32(SUM_ALL, _mm_srli_si128(SUM_ALL, 8));
            sum = sum + _mm_cvtsi128_si32(SUM_ALL);
            
            if (littlesum > sum) {
                (*x) = j;
                (*y) = i;
                littlesum = sum;
            }
        }
    }
    
    return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_bgr(Mat colorImg, Mat subImg, int *x, int *y) {
    if (NULL == colorImg.data || NULL == subImg.data || colorImg.rows < subImg.rows || colorImg.cols < subImg.cols) {
        return SUB_IMAGE_MATCH_FAIL;
    }
    
    uchar *pc = colorImg.data;
    uchar *ps = subImg.data;
    const int rows = colorImg.rows;
    const int cols = colorImg.cols;
    const int subrows = subImg.rows;
    const int subcols = subImg.cols;
    const int diffrows = rows - subrows;
    const int diffcols = cols - subcols;
    int i, j, k, l;
    
    int sum = 0;
    int littlesum = 0x4fffffff;
    (*x) = 0;
    (*y) = 0;
    
    __m128i SAD21, SAD22, SAD23, SAD2, SUM;
    for (i = 0; i <= diffrows; i++) {
        for (j = 0; j <= diffcols; j++) {
            sum = 0;
            SAD2 = ZERO;
            SUM = ZERO;
            for (k = 0; k < subrows; k++) {
                for (l = 0; l < subcols - 16; l += 16) {
                    SAD21 = _mm_sad_epu8(_mm_loadu_si128((__m128i *) (pc + 3 * ((i + k) * cols + j + l) + 0)),
                                         _mm_loadu_si128((__m128i *) (ps + 3 * (k * subcols + l) + 0)));
                    SAD22 = _mm_sad_epu8(_mm_loadu_si128((__m128i *) (pc + 3 * ((i + k) * cols + j + l) + 16)),
                                         _mm_loadu_si128((__m128i *) (ps + 3 * (k * subcols + l) + 16)));
                    SAD23 = _mm_sad_epu8(_mm_loadu_si128((__m128i *) (pc + 3 * ((i + k) * cols + j + l) + 32)),
                                         _mm_loadu_si128((__m128i *) (ps + 3 * (k * subcols + l) + 32)));
                    
                    SAD2 = _mm_add_epi32(_mm_add_epi32(SAD21, SAD22), SAD23);
                    
                    SUM = _mm_add_epi32(SUM, SAD2);
                    
                }
                
                for (; l < subcols; l++) {
                    sum = sum + abs((int) pc[3 * ((i + k) * cols + j + l)] - (int) ps[3 * (k * subcols + l)]) +
                    abs((int) pc[3 * ((i + k) * cols + j + l) + 1] - (int) ps[3 * (k * subcols + l) + 1]) +
                    abs((int) pc[3 * ((i + k) * cols + j + l) + 2] - (int) ps[3 * (k * subcols + l) + 2]);
                }
            }
            
            SUM = _mm_add_epi32(SUM, _mm_srli_si128(SUM, 8));
            sum = sum + _mm_cvtsi128_si32(SUM);
            
            if (littlesum > sum) {
                
                (*x) = j;
                (*y) = i;
                littlesum = sum;
            }
        }
    }
    
    return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_corr(Mat grayImg, Mat subImg, int *x, int *y) {
    if (NULL == grayImg.data || NULL == subImg.data || grayImg.rows < subImg.rows || grayImg.cols < subImg.cols) {
        return SUB_IMAGE_MATCH_FAIL;
    }
    
    uchar *pg = grayImg.data;
    uchar *ps = subImg.data;
    const int rows = grayImg.rows;
    const int cols = grayImg.cols;
    const int subrows = subImg.rows;
    const int subcols = subImg.cols;
    const int diffrows = rows - subrows;
    const int diffcols = cols - subcols;
    int i, j, k, l;
    
    float corr = 0;
    float maxcorr = -1.f;
    (*x) = 0;
    (*y) = 0;
    
    
    __m128i SUMSS = ZERO, SUMTT = ZERO, SUMST = ZERO;
    for (i = 0; i <= diffrows; i++) {
        for (j = 0; j <= diffcols; j++) {
            SUMSS = ZERO;
            SUMTT = ZERO;
            SUMST = ZERO;
            int isumss = 0, isumst = 0, isumtt = 0;
            for (k = 0; k < subrows; k++) {
                for (l = 0; l < subcols - 8; l += 8) {
                    __m128i S16 = (_mm_unpacklo_epi8(_mm_loadu_si128((__m128i *) (pg + (i + k) * cols + j + l)), ZERO));
                    __m128i T16 = (_mm_unpacklo_epi8(_mm_loadu_si128((__m128i *) (ps + k * subcols + l)), ZERO));
                    __m128i SS = _mm_mullo_epi16(S16, S16);
                    __m128i TT = _mm_mullo_epi16(T16, T16);
                    __m128i ST = _mm_mullo_epi16(S16, T16);
                    SUMSS = _mm_add_epi32(_mm_add_epi32(SUMSS, _mm_unpacklo_epi16(SS, ZERO)),
                                          _mm_unpackhi_epi16(SS, ZERO));
                    SUMST = _mm_add_epi32(_mm_add_epi32(SUMST, _mm_unpacklo_epi16(ST, ZERO)),
                                          _mm_unpackhi_epi16(ST, ZERO));
                    SUMTT = _mm_add_epi32(_mm_add_epi32(SUMTT, _mm_unpacklo_epi16(TT, ZERO)),
                                          _mm_unpackhi_epi16(TT, ZERO));
                }
                for (; l < subcols; l++) {
                    isumss += pg[l] * pg[l];
                    isumtt += ps[l] * ps[l];
                    isumst += ps[l] * pg[l];
                }
            }
            SUMSS = _mm_add_epi32(SUMSS, _mm_srli_si128(SUMSS, 8));
            SUMSS = _mm_add_epi32(SUMSS, _mm_srli_si128(SUMSS, 4));
            SUMST = _mm_add_epi32(SUMST, _mm_srli_si128(SUMST, 8));
            SUMST = _mm_add_epi32(SUMST, _mm_srli_si128(SUMST, 4));
            SUMTT = _mm_add_epi32(SUMTT, _mm_srli_si128(SUMTT, 8));
            SUMTT = _mm_add_epi32(SUMTT, _mm_srli_si128(SUMTT, 4));
            float sumss = isumss + _mm_cvtsi128_si32(SUMSS);
            float sumst = isumst + _mm_cvtsi128_si32(SUMST);
            float sumtt = isumtt + _mm_cvtsi128_si32(SUMTT);
            corr = sumst / sqrt(sumss * sumtt);
            if (maxcorr < corr) {
                (*x) = j;
                (*y) = i;
                maxcorr = corr;
            }
        }
    }
    return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_angle(Mat grayImg, Mat subImg, int *x, int *y) {
    if (NULL == grayImg.data || NULL == subImg.data || grayImg.rows < subImg.rows || grayImg.cols < subImg.cols) {
        return SUB_IMAGE_MATCH_FAIL;
    }
    Mat imageX = Mat::zeros(grayImg.size(), CV_32FC1);
    Mat imageY = Mat::zeros(grayImg.size(), CV_32FC1);
    Mat imageAng = Mat::zeros(grayImg.size(), CV_32FC1);
    Mat imageMag = Mat::zeros(grayImg.size(), CV_32FC1);
    ustc_CalcGrad(grayImg, imageX, imageY);
    ustc_CalcAngleMag(imageX, imageY, imageAng, imageMag);
    Mat simageX = Mat::zeros(subImg.size(), CV_32FC1);
    Mat simageY = Mat::zeros(subImg.size(), CV_32FC1);
    Mat simageAng = Mat::zeros(subImg.size(), CV_32FC1);
    Mat simageMag = Mat::zeros(subImg.size(), CV_32FC1);
    ustc_CalcGrad(subImg, simageX, simageY);
    ustc_CalcAngleMag(simageX, simageY, simageAng, simageMag);
    
    float *pg = (float *) imageAng.data;
    float *ps = (float *) simageAng.data;
    const int rows = imageAng.rows;
    const int cols = imageAng.cols;
    const int subrows = simageAng.rows;
    const int subcols = simageAng.cols;
    const int diffrows = rows - subrows;
    const int diffcols = cols - subcols;
    int i, j, k, l;
    
    
    (*x) = 0;
    (*y) = 0;
    
    
    int sum = 0;
    int *tempsum;
    int littlesum = 0x4fffffff;
    __m128i Deg_180 = _mm_set1_epi32(180), Deg_360 = _mm_set1_epi32(360);
    __m128i SUM_all;
    __m128i SAD2, SUM2, SAD2_ABS, SAD2_R, x2_lt_180, x2_gt_180;
    __m128i SAD3, SUM3, SAD3_ABS, SAD3_R, x3_lt_180, x3_gt_180;
    __m128i SAD4, SUM4, SAD4_ABS, SAD4_R, x4_lt_180, x4_gt_180;
    __m128i SAD5, SUM5, SAD5_ABS, SAD5_R, x5_lt_180, x5_gt_180;
    for (i = 0; i <= diffrows; i++) {
        for (j = 0; j <= diffcols; j++) {
            sum = 0;
            SUM2 = ZERO;
            SUM3 = ZERO;
            SUM4 = ZERO;
            SUM5 = ZERO;
            SUM_all = ZERO;
            for (k = 1; k < subrows-1; k++) {
                for (l = 1; l < subcols - 16-1; l += 16) {
                    
                    SAD2 = _mm_cvtps_epi32(_mm_sub_ps(_mm_loadu_ps((pg + (i + k) * cols + j + l)),
                                                      _mm_loadu_ps((ps + k * subcols + l))));
                    SAD3 = _mm_cvtps_epi32(_mm_sub_ps(_mm_loadu_ps((pg + (i + k) * cols + j + l + 4)),
                                                      _mm_loadu_ps((ps + k * subcols + l + 4))));
                    SAD4 = _mm_cvtps_epi32(_mm_sub_ps(_mm_loadu_ps((pg + (i + k) * cols + j + l + 8)),
                                                      _mm_loadu_ps((ps + k * subcols + l + 8))));
                    SAD5 = _mm_cvtps_epi32(_mm_sub_ps(_mm_loadu_ps((pg + (i + k) * cols + j + l + 12)),
                                                      _mm_loadu_ps((ps + k * subcols + l + 12))));
                    
                    SAD2_ABS = _mm_abs_epi32(SAD2);
                    SAD3_ABS = _mm_abs_epi32(SAD3);
                    SAD4_ABS = _mm_abs_epi32(SAD4);
                    SAD5_ABS = _mm_abs_epi32(SAD5);
                    
                    x2_gt_180 = _mm_cmpgt_epi32(SAD2_ABS, Deg_180);
                    x2_lt_180 = _mm_cmplt_epi32(SAD2_ABS, Deg_180);
                    x3_gt_180 = _mm_cmpgt_epi32(SAD3_ABS, Deg_180);
                    x3_lt_180 = _mm_cmplt_epi32(SAD3_ABS, Deg_180);
                    x4_gt_180 = _mm_cmpgt_epi32(SAD4_ABS, Deg_180);
                    x4_lt_180 = _mm_cmplt_epi32(SAD4_ABS, Deg_180);
                    x5_gt_180 = _mm_cmpgt_epi32(SAD5_ABS, Deg_180);
                    x5_lt_180 = _mm_cmplt_epi32(SAD5_ABS, Deg_180);
                    
                    SAD2_R = _mm_add_epi32(_mm_and_si128(_mm_sub_epi32(Deg_360, SAD2_ABS), x2_gt_180),
                                           _mm_and_si128(SAD2_ABS, x2_lt_180));
                    SAD3_R = _mm_add_epi32(_mm_and_si128(_mm_sub_epi32(Deg_360, SAD3_ABS), x3_gt_180),
                                           _mm_and_si128(SAD3_ABS, x3_lt_180));
                    SAD4_R = _mm_add_epi32(_mm_and_si128(_mm_sub_epi32(Deg_360, SAD4_ABS), x4_gt_180),
                                           _mm_and_si128(SAD4_ABS, x4_lt_180));
                    SAD5_R = _mm_add_epi32(_mm_and_si128(_mm_sub_epi32(Deg_360, SAD5_ABS), x5_gt_180),
                                           _mm_and_si128(SAD5_ABS, x5_lt_180));
                    
                    SUM2 = _mm_add_epi32(SUM2, SAD2_R);
                    SUM3 = _mm_add_epi32(SUM3, SAD3_R);
                    SUM4 = _mm_add_epi32(SUM3, SAD4_R);
                    SUM5 = _mm_add_epi32(SUM5, SAD5_R);
                    
                }
                for (; l < subcols-1; l++) {
                    int a = abs((int) pg[(i + k) * cols + j + l] - (int) ps[k * subcols + l]);
                    if (a > 180) a = 360 - a;
                    
                    sum += a;
                }
            }
            SUM_all = _mm_add_epi32(_mm_add_epi32(SUM2, SUM3), _mm_add_epi32(SUM4, SUM5));
            tempsum = (int32_t *) &SUM_all;
            sum = sum + tempsum[0] + tempsum[1] + tempsum[2] + tempsum[3];
            
            if (littlesum > sum) {
                (*x) = j;
                (*y) = i;
                littlesum = sum;
            }
        }
    }
   
    return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_mag(Mat grayImg, Mat subImg, int *x, int *y) {
    if (NULL == grayImg.data || NULL == subImg.data || grayImg.rows < subImg.rows || grayImg.cols < subImg.cols) {
        return SUB_IMAGE_MATCH_FAIL;
    }
    Mat imageX = Mat::zeros(grayImg.size(), CV_32FC1);
    Mat imageY = Mat::zeros(grayImg.size(), CV_32FC1);
    Mat imageAng = Mat::zeros(grayImg.size(), CV_32FC1);
    Mat imageMag = Mat::zeros(grayImg.size(), CV_32FC1);
    ustc_CalcGrad(grayImg, imageX, imageY);
    ustc_CalcAngleMag(imageX, imageY, imageAng, imageMag);
    Mat simageX = Mat::zeros(subImg.size(), CV_32FC1);
    Mat simageY = Mat::zeros(subImg.size(), CV_32FC1);
    Mat simageAng = Mat::zeros(subImg.size(), CV_32FC1);
    Mat simageMag = Mat::zeros(subImg.size(), CV_32FC1);
    ustc_CalcGrad(subImg, simageX, simageY);
    ustc_CalcAngleMag(simageX, simageY, simageAng, simageMag);
    
    float *pg = (float *) imageMag.data;
    float *ps = (float *) simageMag.data;
    const int rows = imageMag.rows;
    const int cols = imageMag.cols;
    const int subrows = simageMag.rows;
    const int subcols = simageMag.cols;
    const int diffrows = rows - subrows;
    const int diffcols = cols - subcols;
    int i, j, k, l;
    
    
    __m128i SAD2, SUM2 = ZERO;
    int sum = 0;
    int *tempsum;
    int littlesum = 0x4fffffff;
    (*x) = 0;
    (*y) = 0;
    
    for (i = 0; i <= diffrows; i++) {
        for (j = 0; j <= diffcols; j++) {
            sum = 0;
            SUM2 = ZERO;
            for (k = 1; k < subrows-1; k++) {
                for (l = 1; l < subcols - 4-1; l += 4) {
                    SAD2 = _mm_cvtps_epi32(_mm_sub_ps(_mm_loadu_ps((pg + (i + k) * cols + j + l)),
                                                      _mm_loadu_ps((ps + k * subcols + l))));
                    SUM2 = _mm_add_epi32(SUM2, _mm_abs_epi32(SAD2));
                }
                for (; l < subcols-1; l++) {
                    sum += abs((int) pg[(i + k) * cols + j + l] - (int) ps[k * subcols + l]);
                }
            }
            tempsum = (int32_t *) &SUM2;
            sum = tempsum[0] + tempsum[1] + tempsum[2] + tempsum[3];
            if (littlesum > sum) {
                (*x) = j;
                (*y) = i;
                littlesum = sum;
            }
        }
    }

    return SUB_IMAGE_MATCH_OK;
}

int ustc_SubImgMatch_hist(Mat grayImg, Mat subImg, int *x, int *y) {
    if (NULL == grayImg.data || NULL == subImg.data || grayImg.rows < subImg.rows || grayImg.cols < subImg.cols) {
        return SUB_IMAGE_MATCH_FAIL;
    }
    uchar *pg = grayImg.data;
    uchar *ps = subImg.data;
    const int rows = grayImg.rows;
    const int cols = grayImg.cols;
    const int subrows = subImg.rows;
    const int subcols = subImg.cols;
    const int diffrows = rows - subrows;
    const int diffcols = cols - subcols;
    int i, j, k, l;
    
    int diffhist[256];
    int hist_len = 256;
    
    int sum = 0;
    int littlesum = 0x4fffffff;
    (*x) = 0;
    (*y) = 0;
    
    
    __m128i SUM1, SUM2, SUM3, SUM4, SUM_ALL;
    for (i = 0; i <= diffrows; ++i) {
        for (j = 0; j <= diffcols; ++j) {
            sum = 0;
            SUM_ALL = ZERO;
            memset(diffhist, 0, sizeof(int) * hist_len);
            for (k = 0; k < subrows; ++k) {
                for (l = 0; l < subcols - 4; l += 4) {
                    diffhist[pg[(i + k) * cols + j + l]] += 1;
                    diffhist[pg[(i + k) * cols + j + l + 1]] += 1;
                    diffhist[pg[(i + k) * cols + j + l + 2]] += 1;
                    diffhist[pg[(i + k) * cols + j + l + 3]] += 1;
                    
                    diffhist[ps[k * subcols + l]] -= 1;
                    diffhist[ps[k * subcols + l + 1]] -= 1;
                    diffhist[ps[k * subcols + l + 2]] -= 1;
                    diffhist[ps[k * subcols + l + 3]] -= 1;
                    
                    
                }
                for (; l < subcols; l++) {
                    
                    diffhist[pg[(i + k) * cols + j + l]] += 1;
                    diffhist[ps[k * subcols + l]] -= 1;
                }
            }
            
            for (int ii = 0; ii < hist_len; ii += 16) {
                
                SUM1 = _mm_abs_epi32(_mm_load_si128((__m128i *) (diffhist + ii)));
                SUM2 = _mm_abs_epi32(_mm_load_si128((__m128i *) (diffhist + ii + 4)));
                SUM3 = _mm_abs_epi32(_mm_load_si128((__m128i *) (diffhist + ii + 8)));
                SUM4 = _mm_abs_epi32(_mm_load_si128((__m128i *) (diffhist + ii + 12)));
                SUM_ALL = _mm_add_epi32(SUM_ALL, SUM4);
            }
            SUM_ALL = _mm_add_epi32(SUM_ALL, _mm_srli_si128(SUM_ALL, 8));
            SUM_ALL = _mm_add_epi32(SUM_ALL, _mm_srli_si128(SUM_ALL, 4));
            sum = _mm_cvtsi128_si32(SUM_ALL);
            if (littlesum > sum) {
                (*x) = j;
                (*y) = i;
                littlesum = sum;
                
            }
        }
    }
    return SUB_IMAGE_MATCH_OK;
}
