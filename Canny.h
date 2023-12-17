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
    Mat src = _src.getMat();           //输入图像，必须为单通道灰度图  
    CV_Assert(src.depth() == CV_8U); // 8位无符号  

    _dst.create(src.size(), CV_8U);    //根据src的大小构造目标矩阵dst  
    Mat dst = _dst.getMat();           //输出图像，为单通道黑白图  


    // low_thresh 表示低阈值， high_thresh表示高阈值  
    // aperture_size 表示算子大小，默认为3  
    // L2gradient计算梯度幅值的标识，默认为false  

    // 如果L2gradient为false 并且 apeture_size的值为-1（-1的二进制标识为：1111 1111）  
    // L2gradient为false 则计算sobel导数时，用G = |Gx|+|Gy|  
    // L2gradient为true  则计算sobel导数时，用G = Math.sqrt((Gx)^2 + (Gy)^2) 根号下 开平方  

    if (!L2gradient && (aperture_size & CV_CANNY_L2_GRADIENT) == CV_CANNY_L2_GRADIENT)
    {
        // CV_CANNY_L2_GRADIENT 宏定义其值为： Value = (1<<31) 1左移31位  即2147483648  
            //backward compatibility  

        // ~标识按位取反  
        aperture_size &= ~CV_CANNY_L2_GRADIENT;//相当于取绝对值  
        L2gradient = true;
    }


    // 判别条件1：aperture_size是奇数  
    // 判别条件2: aperture_size的范围应当是[3,7], 默认值3   
    if ((aperture_size & 1) == 0 || (aperture_size != -1 && (aperture_size < 3 || aperture_size > 7)))
        CV_Error(Error::StsBadArg, "");  // 报错  

    if (low_thresh > high_thresh)           // 如果低阈值 > 高阈值  
        std::swap(low_thresh, high_thresh); // 则交换低阈值和高阈值  

    const int cn = src.channels();           // cn为输入图像的通道数  
    Mat dx(src.rows, src.cols, CV_16SC(cn)); // 存储 x方向 方向导数的矩阵，CV_16SC(cn)：16位有符号cn通道  
    Mat dy(src.rows, src.cols, CV_16SC(cn)); // 存储 y方向 方向导数的矩阵 ......  

    /*Sobel参数说明：(参考cvSobel)
      cvSobel(
            const  CvArr* src,                // 输入图像
            CvArr*        dst,                // 输入图像
            int           xorder，            // x方向求导的阶数
            int           yorder，         // y方向求导的阶数
            int           aperture_size = 3   // 滤波器的宽和高 必须是奇数
      );
    */

    // BORDER_REPLICATE 表示当卷积点在图像的边界时，原始图像边缘的像素会被复制，并用复制的像素扩展原始图的尺寸  
    // 计算x方向的sobel方向导数，计算结果存在dx中  
    Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0, cv::BORDER_REPLICATE);
    // 计算y方向的sobel方向导数，计算结果存在dy中  
    Sobel(src, dy, CV_16S, 0, 1, aperture_size, 1, 0, cv::BORDER_REPLICATE);

    //L2gradient为true时， 表示需要根号下开平方运算，阈值也需要平方  
    if (L2gradient)
    {
        low_thresh = std::min(32767.0, low_thresh);
        high_thresh = std::min(32767.0, high_thresh);

        if (low_thresh > 0) low_thresh *= low_thresh;    //低阈值平方运算  
        if (high_thresh > 0) high_thresh *= high_thresh; //高阈值平方运算  
    }

    int low = cvFloor(low_thresh);   // cvFloor返回不大于参数的最大整数值, 相当于取整  
    int high = cvFloor(high_thresh);

    // ptrdiff_t 是C/C++标准库中定义的一个数据类型，signed类型，通常用于存储两个指针的差（距离），可以是负数  
    // mapstep 用于存放  
    ptrdiff_t mapstep = src.cols + 2; // +2 表示左右各扩展一条边  

    // AutoBuffer<uchar> 会自动分配一定大小的内存，并且指定内存中的数据类型是uchar  
    // 列数 +2 表示图像左右各自扩展一条边 （用于复制边缘像素，扩大原始图像）  
    // 行数 +2 表示图像上下各自扩展一条边  
    AutoBuffer<uchar> buffer((src.cols + 2) * (src.rows + 2) + cn * mapstep * 3 * sizeof(int));

    int* mag_buf[3];  //定义一个大小为3的int型指针数组，  
    mag_buf[0] = (int*)(uchar*)buffer;
    mag_buf[1] = mag_buf[0] + mapstep * cn;
    mag_buf[2] = mag_buf[1] + mapstep * cn;
    memset(mag_buf[0], 0, /* cn* */mapstep * sizeof(int));

    uchar* map = (uchar*)(mag_buf[2] + mapstep * cn);
    memset(map, 1, mapstep);
    memset(map + mapstep * (src.rows + 1), 1, mapstep);

    int maxsize = std::max(1 << 10, src.cols * src.rows / 10); // 2的10次幂 1024  
    std::vector<uchar*> stack(maxsize); // 定义指针类型向量，用于存地址  
    uchar** stack_top = &stack[0];      // 栈顶指针（指向指针的指针），指向stack[0], stack[0]也是一个指针  
    uchar** stack_bottom = &stack[0];   // 栈底指针 ，初始时 栈底指针 == 栈顶指针  


    // 梯度的方向被近似到四个角度之一 (0, 45, 90, 135 四选一)  
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


    // define 定义函数块  
    // CANNY_PUSH(d) 是入栈函数， 参数d表示地址指针，让该指针指向的内容为2（int型强制转换成uchar型），并入栈，栈顶指针+1  
    // 2表示 像素属于某条边缘 可以看下方的注释  
    // CANNY_POP(d) 是出栈函数， 栈顶指针-1，然后将-1后的栈顶指针指向的值，赋给d  
#define CANNY_PUSH(d)    *(d) = uchar(2), *stack_top++ = (d)  
#define CANNY_POP(d)     (d) = *--stack_top  

// calculate magnitude and angle of gradient, perform non-maxima suppression.  
// fill the map with one of the following values:  
// 0 - the pixel might belong to an edge 可能属于边缘  
// 1 - the pixel can not belong to an edge 不属于边缘  
// 2 - the pixel does belong to an edge 一定属于边缘  

// for内进行非极大值抑制 + 滞后阈值处理  
    for (int i = 0; i <= src.rows; i++) // i 表示第i行  
    {

        // i == 0 时，_norm 指向 mag_buf[1]  
        // i > 0 时， _norm 指向 mag_buf[2]  
        // +1 表示跳过每行的第一个元素，因为是后扩展的边，不可能是边缘  
        int* _norm = mag_buf[(i > 0) + 1] + 1;

        if (i < src.rows)
        {
            short* _dx = dx.ptr<short>(i); // _dx指向dx矩阵的第i行  
            short* _dy = dy.ptr<short>(i); // _dy指向dy矩阵的第i行  

            if (!L2gradient) // 如果 L2gradient为false  
            {
                for (int j = 0; j < src.cols * cn; j++) // 对第i行里的每一个值都进行计算  
                    _norm[j] = std::abs(int(_dx[j])) + std::abs(int(_dy[j])); // 用||+||计算  
            }
            else
            {
                for (int j = 0; j < src.cols * cn; j++)
                    //用平方计算,当 L2gradient为 true时，高低阈值都被平方了，所以此处_norm[j]无需开平方  
                    _norm[j] = int(_dx[j]) * _dx[j] + int(_dy[j]) * _dy[j]; //  
            }

            if (cn > 1) // 如果不是单通道  
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
            _norm[-1] = _norm[src.cols] = 0; // 最后一列和第一列的梯度幅值设置为0  
        }
        // 当i == src.rows （最后一行）时，申请空间并且每个空间的值初始化为0, 存储在mag_buf[2]中  
        else
            memset(_norm - 1, 0, /* cn* */mapstep * sizeof(int));

        // at the very beginning we do not have a complete ring  
        // buffer of 3 magnitude rows for non-maxima suppression  
        if (i == 0)
            continue;

        uchar* _map = map + mapstep * i + 1; // _map 指向第 i+1 行，+1表示跳过该行第一个元素  
        _map[-1] = _map[src.cols] = 1; // 第一列和最后一列不是边缘，所以设置为1  

        int* _mag = mag_buf[1] + 1; // take the central row 中间那一行  
        ptrdiff_t magstep1 = mag_buf[2] - mag_buf[1];
        ptrdiff_t magstep2 = mag_buf[0] - mag_buf[1];

        const short* _x = dx.ptr<short>(i - 1);
        const short* _y = dy.ptr<short>(i - 1);

        // 如果栈的大小不够，则重新为栈分配内存（相当于扩大容量）  
        if ((stack_top - stack_bottom) + src.cols > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = maxsize * 3 / 2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        int prev_flag = 0; //前一个像素点 0：非边缘点 ；1：边缘点  
        for (int j = 0; j < src.cols; j++) // 第 j 列  
        {
#define CANNY_SHIFT 15  
            // tan22.5  
            const int TG22 = (int)(0.4142135623730950488016887242097 * (1 << CANNY_SHIFT) + 0.5);

            int m = _mag[j];

            if (m > low) // 如果大于低阈值  
            {
                int xs = _x[j];    // dx中 第i-1行 第j列  
                int ys = _y[j];    // dy中 第i-1行 第j列  
                int x = std::abs(xs);
                int y = std::abs(ys) << CANNY_SHIFT;

                int tg22x = x * TG22;

                if (y < tg22x) //角度小于22.5 用区间表示：[0, 22.5)  
                {
                    // 与左右两点的梯度幅值比较，如果比左右都大  
                    //（此时当前点是左右邻域内的极大值），则 goto __ocv_canny_push 执行入栈操作  
                    if (m > _mag[j - 1] && m >= _mag[j + 1]) goto __ocv_canny_push;
                }
                else //角度大于22.5  
                {
                    int tg67x = tg22x + (x << (CANNY_SHIFT + 1));
                    if (y > tg67x) //(67.5, 90)  
                    {
                        //与上下两点的梯度幅值比较，如果比上下都大  
                        //（此时当前点是左右邻域内的极大值），则 goto __ocv_canny_push 执行入栈操作  
                        if (m > _mag[j + magstep2] && m >= _mag[j + magstep1]) goto __ocv_canny_push;
                    }
                    else //[22.5, 67.5]  
                    {
                        // ^ 按位异或 如果xs与ys异号 则取-1 否则取1  
                        int s = (xs ^ ys) < 0 ? -1 : 1;
                        //比较对角线邻域  
                        if (m > _mag[j + magstep2 - s] && m > _mag[j + magstep1 + s]) goto __ocv_canny_push;
                    }
                }
            }

            //比当前的梯度幅值低阈值还低，直接被确定为非边缘  
            prev_flag = 0;
            _map[j] = uchar(1); // 1 表示不属于边缘  

            continue;
        __ocv_canny_push:
            // 前一个点不是边缘点 并且 当前点的幅值大于高阈值（大于高阈值被视为边缘像素） 并且 正上方的点不是边缘点  
            if (!prev_flag && m > high && _map[j - mapstep] != 2)
            {
                //将当前点的地址入栈，入栈前，会将该点地址指向的值设置为2（查看上面的宏定义函数块里）  
                CANNY_PUSH(_map + j);
                prev_flag = 1;
            }
            else
                _map[j] = 0;
        }

        // scroll the ring buffer  
        // 交换指针指向的位置，向上覆盖，把mag_[1]的内容覆盖到mag_buf[0]上  
        // 把mag_[2]的内容覆盖到mag_buf[1]上  
    // 最后 让mag_buf[2]指向_mag指向的那一行  
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
    }


    // now track the edges (hysteresis thresholding)  
    // 通过上面的for循环，确定了各个邻域内的极大值点为边缘点（标记为2）  
    // 现在，在这些边缘点的8邻域内（上下左右+4个对角）,将可能的边缘点（标记为0）确定为边缘  
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

        CANNY_POP(m); // 出栈  

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
    // 生成边缘图  
    const uchar* pmap = map + mapstep + 1;
    uchar* pdst = dst.ptr();
    for (int i = 0; i < src.rows; i++, pmap += mapstep, pdst += dst.step)
    {
        for (int j = 0; j < src.cols; j++)
            pdst[j] = (uchar)-(pmap[j] >> 1);
    }
}

