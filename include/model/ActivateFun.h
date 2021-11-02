//
// Created by ltc on 2021/9/10.
//

#ifndef TINYDL_ACTIVATEFUN_H
#define TINYDL_ACTIVATEFUN_H

#include "Tensor.h"
#include "operator/ReluOperator.h"

Tensor sigmoid(const Tensor& t);
Tensor tanh(const Tensor& t);
Tensor relu(const Tensor& t);

#endif //TINYDL_ACTIVATEFUN_H
