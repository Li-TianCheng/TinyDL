//
// Created by ltc on 2021/9/13.
//

#ifndef TINYDL_UTILS_H
#define TINYDL_UTILS_H

#include "Tensor.h"

Tensor softmax(const Tensor& t);
Tensor maxPool(const Tensor& t, int kernelRow, int kernelCol, int stride=1);

#endif //TINYDL_UTILS_H
