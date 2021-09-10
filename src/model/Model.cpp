//
// Created by ltc on 2021/9/9.
//

#include "model/Model.h"

Tensor Model::operator()(const Tensor &input) {
	return forward(input);
}
