#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "kernel_configs.h"

static double TimeSpecToSeconds(struct timespec *ts)
{
    return (double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0;
}

// Fills image with running counter of data
// Does not include padding
void fillImage_floattype(FLOATTYPE *image, int c, int h, int w)
{
    for (int i = 0; i < c; i++)
    {
        for (int j = 0; j < h; j++)
        {
            for (int k = 0; k < w; k++)
            {
                image[i * h * w + j * w + k] = i * h * w + j * w + k;
            }
        }
    }
}

void print_tensor(FLOATTYPE *image, int c, int h, int w)
{
    for (int i = 0; i < c; i++)
    {
        for (int j = 0; j < h; j++)
        {
            for (int k = 0; k < w; k++)
            {
                std::cout << image[i * h * w + j * w + k] << "\t";
            }
            std::cout << "\n";
        }
    }
}

FLOATTYPE calculateChecksum_float(FLOATTYPE *image, int c, int h, int w)
{
    FLOATTYPE checksum = 0.0;
    for (int i = 0; i < c; i++)
    {
        for (int j = 0; j < h; j++)
        {
            for (int k = 0; k < w; k++)
            {
                checksum += image[i * h * w + j * w + k];
            }
        }
    }

    return checksum;
}