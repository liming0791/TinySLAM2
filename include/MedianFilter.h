#ifndef MEDIANFILTER_H
#define MEDIANFILTER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>

template <int S = 5> 
class MedianFilter
{
    private:
        double *data;
        double *buffer;
        int pIdx;
        int winSize;

    public:
        MedianFilter();
        ~MedianFilter();
        double filterAdd(double val);

};

template<int S>
MedianFilter<S>::MedianFilter():pIdx(0), winSize(S)
{
    if ( S <= 0 )
        printf("size can not be 0 or neg !");
    else {
        data = (double *)malloc(sizeof(double)*S);
        buffer = (double *)malloc(sizeof(double)*S);
    }
}

template<int S>
MedianFilter<S>::~MedianFilter()
{
    if (data != NULL)
        free(data);
    if (buffer != NULL)
        free(buffer);
}

template<int S>
double MedianFilter<S>::filterAdd(double val)
{
    data[pIdx] = val;
    pIdx++;
    if (pIdx == winSize)
        pIdx = 0;
    
    // find mid
    memcpy(buffer, data, sizeof(double)*winSize);

    for (int i = 0; i <= winSize/2; i++) {
        int maxIdx = i;
        for (int j = i; j < winSize; j++) {
            if (buffer[j] > buffer[maxIdx]) {
                maxIdx = j;
            }
        }
        double tmp = buffer[i];
        buffer[i] = buffer[maxIdx];
        buffer[maxIdx] = tmp;
    }

    return buffer[winSize/2];
}
#endif
