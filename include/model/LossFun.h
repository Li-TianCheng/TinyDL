//
// Created by ltc on 2021/9/10.
//

#ifndef TINYDL_LOSSFUN_H
#define TINYDL_LOSSFUN_H

#include "Tensor.h"
#include "model/UtilsFun.h"

Tensor MSELoss(Tensor& pred, Tensor& label);
Tensor crossEntropyLoss(Tensor& pred, Tensor& label);

#endif //TINYDL_LOSSFUN_H
