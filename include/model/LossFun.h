//
// Created by ltc on 2021/9/10.
//

#ifndef TINYDL_LOSSFUN_H
#define TINYDL_LOSSFUN_H

#include "Tensor.h"

Tensor MSELoss(const Tensor& pred, const Tensor& label);
Tensor crossEntropyLoss(const Tensor& pred, const Tensor& label);

#endif //TINYDL_LOSSFUN_H
