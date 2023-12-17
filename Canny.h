#pragma once
#include <opencv2/opencv.hpp>

#define CV_CANNY_L2_GRADIENT 2147483648
/*
#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
#define USE_IPP_CANNY 1
#else
#undef USE_IPP_CANNY
#endif
*/

void cv::Canny(InputArray _src, OutputArray _dst,
    double low_thresh, double high_thresh,
    int aperture_size, bool L2gradient)
{
    Mat src = _src.getMat();           //����ͼ�񣬱���Ϊ��ͨ���Ҷ�ͼ  
    CV_Assert(src.depth() == CV_8U); // 8λ�޷���  

    _dst.create(src.size(), CV_8U);    //����src�Ĵ�С����Ŀ�����dst  
    Mat dst = _dst.getMat();           //���ͼ��Ϊ��ͨ���ڰ�ͼ  


    // low_thresh ��ʾ����ֵ�� high_thresh��ʾ����ֵ  
    // aperture_size ��ʾ���Ӵ�С��Ĭ��Ϊ3  
    // L2gradient�����ݶȷ�ֵ�ı�ʶ��Ĭ��Ϊfalse  

    // ���L2gradientΪfalse ���� apeture_size��ֵΪ-1��-1�Ķ����Ʊ�ʶΪ��1111 1111��  
    // L2gradientΪfalse �����sobel����ʱ����G = |Gx|+|Gy|  
    // L2gradientΪtrue  �����sobel����ʱ����G = Math.sqrt((Gx)^2 + (Gy)^2) ������ ��ƽ��  

    if (!L2gradient && (aperture_size & CV_CANNY_L2_GRADIENT) == CV_CANNY_L2_GRADIENT)
    {
        // CV_CANNY_L2_GRADIENT �궨����ֵΪ�� Value = (1<<31) 1����31λ  ��2147483648  
            //backward compatibility  

        // ~��ʶ��λȡ��  
        aperture_size &= ~CV_CANNY_L2_GRADIENT;//�൱��ȡ����ֵ  
        L2gradient = true;
    }


    // �б�����1��aperture_size������  
    // �б�����2: aperture_size�ķ�ΧӦ����[3,7], Ĭ��ֵ3   
    if ((aperture_size & 1) == 0 || (aperture_size != -1 && (aperture_size < 3 || aperture_size > 7)))
        CV_Error(Error::StsBadArg, "");  // ����  

    if (low_thresh > high_thresh)           // �������ֵ > ����ֵ  
        std::swap(low_thresh, high_thresh); // �򽻻�����ֵ�͸���ֵ  

    const int cn = src.channels();           // cnΪ����ͼ���ͨ����  
    Mat dx(src.rows, src.cols, CV_16SC(cn)); // �洢 x���� �������ľ���CV_16SC(cn)��16λ�з���cnͨ��  
    Mat dy(src.rows, src.cols, CV_16SC(cn)); // �洢 y���� �������ľ��� ......  

    /*Sobel����˵����(�ο�cvSobel)
      cvSobel(
            const  CvArr* src,                // ����ͼ��
            CvArr*        dst,                // ����ͼ��
            int           xorder��            // x�����󵼵Ľ���
            int           yorder��         // y�����󵼵Ľ���
            int           aperture_size = 3   // �˲����Ŀ�͸� ����������
      );
    */

    // BORDER_REPLICATE ��ʾ���������ͼ��ı߽�ʱ��ԭʼͼ���Ե�����ػᱻ���ƣ����ø��Ƶ�������չԭʼͼ�ĳߴ�  
    // ����x�����sobel������������������dx��  
    Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0, cv::BORDER_REPLICATE);
    // ����y�����sobel������������������dy��  
    Sobel(src, dy, CV_16S, 0, 1, aperture_size, 1, 0, cv::BORDER_REPLICATE);

    //L2gradientΪtrueʱ�� ��ʾ��Ҫ�����¿�ƽ�����㣬��ֵҲ��Ҫƽ��  
    if (L2gradient)
    {
        low_thresh = std::min(32767.0, low_thresh);
        high_thresh = std::min(32767.0, high_thresh);

        if (low_thresh > 0) low_thresh *= low_thresh;    //����ֵƽ������  
        if (high_thresh > 0) high_thresh *= high_thresh; //����ֵƽ������  
    }

    int low = cvFloor(low_thresh);   // cvFloor���ز����ڲ������������ֵ, �൱��ȡ��  
    int high = cvFloor(high_thresh);

    // ptrdiff_t ��C/C++��׼���ж����һ���������ͣ�signed���ͣ�ͨ�����ڴ洢����ָ��Ĳ���룩�������Ǹ���  
    // mapstep ���ڴ��  
    ptrdiff_t mapstep = src.cols + 2; // +2 ��ʾ���Ҹ���չһ����  

    // AutoBuffer<uchar> ���Զ�����һ����С���ڴ棬����ָ���ڴ��е�����������uchar  
    // ���� +2 ��ʾͼ�����Ҹ�����չһ���� �����ڸ��Ʊ�Ե���أ�����ԭʼͼ��  
    // ���� +2 ��ʾͼ�����¸�����չһ����  
    AutoBuffer<uchar> buffer((src.cols + 2) * (src.rows + 2) + cn * mapstep * 3 * sizeof(int));

    int* mag_buf[3];  //����һ����СΪ3��int��ָ�����飬  
    mag_buf[0] = (int*)(uchar*)buffer;
    mag_buf[1] = mag_buf[0] + mapstep * cn;
    mag_buf[2] = mag_buf[1] + mapstep * cn;
    memset(mag_buf[0], 0, /* cn* */mapstep * sizeof(int));

    uchar* map = (uchar*)(mag_buf[2] + mapstep * cn);
    memset(map, 1, mapstep);
    memset(map + mapstep * (src.rows + 1), 1, mapstep);

    int maxsize = std::max(1 << 10, src.cols * src.rows / 10); // 2��10���� 1024  
    std::vector<uchar*> stack(maxsize); // ����ָ���������������ڴ��ַ  
    uchar** stack_top = &stack[0];      // ջ��ָ�루ָ��ָ���ָ�룩��ָ��stack[0], stack[0]Ҳ��һ��ָ��  
    uchar** stack_bottom = &stack[0];   // ջ��ָ�� ����ʼʱ ջ��ָ�� == ջ��ָ��  


    // �ݶȵķ��򱻽��Ƶ��ĸ��Ƕ�֮һ (0, 45, 90, 135 ��ѡһ)  
    /* sector numbers
       (Top-Left Origin)

        1   2   3
         *  *  *
          * * *
        0*******0
          * * *
         *  *  *
        3   2   1
    */


    // define ���庯����  
    // CANNY_PUSH(d) ����ջ������ ����d��ʾ��ַָ�룬�ø�ָ��ָ�������Ϊ2��int��ǿ��ת����uchar�ͣ�������ջ��ջ��ָ��+1  
    // 2��ʾ ��������ĳ����Ե ���Կ��·���ע��  
    // CANNY_POP(d) �ǳ�ջ������ ջ��ָ��-1��Ȼ��-1���ջ��ָ��ָ���ֵ������d  
#define CANNY_PUSH(d)    *(d) = uchar(2), *stack_top++ = (d)  
#define CANNY_POP(d)     (d) = *--stack_top  

// calculate magnitude and angle of gradient, perform non-maxima suppression.  
// fill the map with one of the following values:  
// 0 - the pixel might belong to an edge �������ڱ�Ե  
// 1 - the pixel can not belong to an edge �����ڱ�Ե  
// 2 - the pixel does belong to an edge һ�����ڱ�Ե  

// for�ڽ��зǼ���ֵ���� + �ͺ���ֵ����  
    for (int i = 0; i <= src.rows; i++) // i ��ʾ��i��  
    {

        // i == 0 ʱ��_norm ָ�� mag_buf[1]  
        // i > 0 ʱ�� _norm ָ�� mag_buf[2]  
        // +1 ��ʾ����ÿ�еĵ�һ��Ԫ�أ���Ϊ�Ǻ���չ�ıߣ��������Ǳ�Ե  
        int* _norm = mag_buf[(i > 0) + 1] + 1;

        if (i < src.rows)
        {
            short* _dx = dx.ptr<short>(i); // _dxָ��dx����ĵ�i��  
            short* _dy = dy.ptr<short>(i); // _dyָ��dy����ĵ�i��  

            if (!L2gradient) // ��� L2gradientΪfalse  
            {
                for (int j = 0; j < src.cols * cn; j++) // �Ե�i�����ÿһ��ֵ�����м���  
                    _norm[j] = std::abs(int(_dx[j])) + std::abs(int(_dy[j])); // ��||+||����  
            }
            else
            {
                for (int j = 0; j < src.cols * cn; j++)
                    //��ƽ������,�� L2gradientΪ trueʱ���ߵ���ֵ����ƽ���ˣ����Դ˴�_norm[j]���迪ƽ��  
                    _norm[j] = int(_dx[j]) * _dx[j] + int(_dy[j]) * _dy[j]; //  
            }

            if (cn > 1) // ������ǵ�ͨ��  
            {
                for (int j = 0, jn = 0; j < src.cols; ++j, jn += cn)
                {
                    int maxIdx = jn;
                    for (int k = 1; k < cn; ++k)
                        if (_norm[jn + k] > _norm[maxIdx]) maxIdx = jn + k;
                    _norm[j] = _norm[maxIdx];
                    _dx[j] = _dx[maxIdx];
                    _dy[j] = _dy[maxIdx];
                }
            }
            _norm[-1] = _norm[src.cols] = 0; // ���һ�к͵�һ�е��ݶȷ�ֵ����Ϊ0  
        }
        // ��i == src.rows �����һ�У�ʱ������ռ䲢��ÿ���ռ��ֵ��ʼ��Ϊ0, �洢��mag_buf[2]��  
        else
            memset(_norm - 1, 0, /* cn* */mapstep * sizeof(int));

        // at the very beginning we do not have a complete ring  
        // buffer of 3 magnitude rows for non-maxima suppression  
        if (i == 0)
            continue;

        uchar* _map = map + mapstep * i + 1; // _map ָ��� i+1 �У�+1��ʾ�������е�һ��Ԫ��  
        _map[-1] = _map[src.cols] = 1; // ��һ�к����һ�в��Ǳ�Ե����������Ϊ1  

        int* _mag = mag_buf[1] + 1; // take the central row �м���һ��  
        ptrdiff_t magstep1 = mag_buf[2] - mag_buf[1];
        ptrdiff_t magstep2 = mag_buf[0] - mag_buf[1];

        const short* _x = dx.ptr<short>(i - 1);
        const short* _y = dy.ptr<short>(i - 1);

        // ���ջ�Ĵ�С������������Ϊջ�����ڴ棨�൱������������  
        if ((stack_top - stack_bottom) + src.cols > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = maxsize * 3 / 2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        int prev_flag = 0; //ǰһ�����ص� 0���Ǳ�Ե�� ��1����Ե��  
        for (int j = 0; j < src.cols; j++) // �� j ��  
        {
#define CANNY_SHIFT 15  
            // tan22.5  
            const int TG22 = (int)(0.4142135623730950488016887242097 * (1 << CANNY_SHIFT) + 0.5);

            int m = _mag[j];

            if (m > low) // ������ڵ���ֵ  
            {
                int xs = _x[j];    // dx�� ��i-1�� ��j��  
                int ys = _y[j];    // dy�� ��i-1�� ��j��  
                int x = std::abs(xs);
                int y = std::abs(ys) << CANNY_SHIFT;

                int tg22x = x * TG22;

                if (y < tg22x) //�Ƕ�С��22.5 �������ʾ��[0, 22.5)  
                {
                    // ������������ݶȷ�ֵ�Ƚϣ���������Ҷ���  
                    //����ʱ��ǰ�������������ڵļ���ֵ������ goto __ocv_canny_push ִ����ջ����  
                    if (m > _mag[j - 1] && m >= _mag[j + 1]) goto __ocv_canny_push;
                }
                else //�Ƕȴ���22.5  
                {
                    int tg67x = tg22x + (x << (CANNY_SHIFT + 1));
                    if (y > tg67x) //(67.5, 90)  
                    {
                        //������������ݶȷ�ֵ�Ƚϣ���������¶���  
                        //����ʱ��ǰ�������������ڵļ���ֵ������ goto __ocv_canny_push ִ����ջ����  
                        if (m > _mag[j + magstep2] && m >= _mag[j + magstep1]) goto __ocv_canny_push;
                    }
                    else //[22.5, 67.5]  
                    {
                        // ^ ��λ��� ���xs��ys��� ��ȡ-1 ����ȡ1  
                        int s = (xs ^ ys) < 0 ? -1 : 1;
                        //�Ƚ϶Խ�������  
                        if (m > _mag[j + magstep2 - s] && m > _mag[j + magstep1 + s]) goto __ocv_canny_push;
                    }
                }
            }

            //�ȵ�ǰ���ݶȷ�ֵ����ֵ���ͣ�ֱ�ӱ�ȷ��Ϊ�Ǳ�Ե  
            prev_flag = 0;
            _map[j] = uchar(1); // 1 ��ʾ�����ڱ�Ե  

            continue;
        __ocv_canny_push:
            // ǰһ���㲻�Ǳ�Ե�� ���� ��ǰ��ķ�ֵ���ڸ���ֵ�����ڸ���ֵ����Ϊ��Ե���أ� ���� ���Ϸ��ĵ㲻�Ǳ�Ե��  
            if (!prev_flag && m > high && _map[j - mapstep] != 2)
            {
                //����ǰ��ĵ�ַ��ջ����ջǰ���Ὣ�õ��ַָ���ֵ����Ϊ2���鿴����ĺ궨�庯�����  
                CANNY_PUSH(_map + j);
                prev_flag = 1;
            }
            else
                _map[j] = 0;
        }

        // scroll the ring buffer  
        // ����ָ��ָ���λ�ã����ϸ��ǣ���mag_[1]�����ݸ��ǵ�mag_buf[0]��  
        // ��mag_[2]�����ݸ��ǵ�mag_buf[1]��  
    // ��� ��mag_buf[2]ָ��_magָ�����һ��  
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
    }


    // now track the edges (hysteresis thresholding)  
    // ͨ�������forѭ����ȷ���˸��������ڵļ���ֵ��Ϊ��Ե�㣨���Ϊ2��  
    // ���ڣ�����Щ��Ե���8�����ڣ���������+4���Խǣ�,�����ܵı�Ե�㣨���Ϊ0��ȷ��Ϊ��Ե  
    while (stack_top > stack_bottom)
    {
        uchar* m;
        if ((stack_top - stack_bottom) + 8 > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = maxsize * 3 / 2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        CANNY_POP(m); // ��ջ  

        if (!m[-1])         CANNY_PUSH(m - 1);
        if (!m[1])          CANNY_PUSH(m + 1);
        if (!m[-mapstep - 1]) CANNY_PUSH(m - mapstep - 1);
        if (!m[-mapstep])   CANNY_PUSH(m - mapstep);
        if (!m[-mapstep + 1]) CANNY_PUSH(m - mapstep + 1);
        if (!m[mapstep - 1])  CANNY_PUSH(m + mapstep - 1);
        if (!m[mapstep])    CANNY_PUSH(m + mapstep);
        if (!m[mapstep + 1])  CANNY_PUSH(m + mapstep + 1);
    }

    // the final pass, form the final image  
    // ���ɱ�Եͼ  
    const uchar* pmap = map + mapstep + 1;
    uchar* pdst = dst.ptr();
    for (int i = 0; i < src.rows; i++, pmap += mapstep, pdst += dst.step)
    {
        for (int j = 0; j < src.cols; j++)
            pdst[j] = (uchar)-(pmap[j] >> 1);
    }
}

