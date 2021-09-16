//
// Created by ltc on 2021/9/13.
//

#ifndef TINYDL_UTILSFUN_H
#define TINYDL_UTILSFUN_H

#include "Tensor.h"

Tensor softmax(const Tensor& t);
Tensor maxPool(const Tensor& t, int channel, int dataRow, int dataCol, int kernelRow, int kernelCol, int stride=1);

#endif //TINYDL_UTILSFUN_H
