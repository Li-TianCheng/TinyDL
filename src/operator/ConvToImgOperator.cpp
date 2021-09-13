//
// Created by ltc on 2021/9/13.
//

#include "operator/ConvToImgOperator.h"

ConvToImgOperator::ConvToImgOperator(const Tensor &tensor1, int channel) : Operator(tensor1, 0), channel(channel) {

}

Tensor ConvToImgOperator::operator()() {
	auto value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>();
	value->resize(tensor1.row()/channel, tensor1.col()*channel);
	for (int i = 0; i < tensor1.row(); ++i) {
		for (int j = 0; j < tensor1.col(); ++j) {
			(*value)(i/channel, j+i%channel*tensor1.col()) = tensor1(i, j);
		}
	}
	return Tensor(value, shared_from_this());
}

void ConvToImgOperator::backward(Tensor &result) {
	for (int i = 0; i < tensor1.row(); ++i) {
		for (int j = 0; j < tensor1.col(); ++j) {
			tensor1.grad()(i, j) += result.grad()(i/channel, j+i%channel*tensor1.col());
		}
	}
}
