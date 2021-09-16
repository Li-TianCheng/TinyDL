//
// Created by ltc on 2021/9/13.
//

#include "model/UtilsFun.h"
#include "operator/MaxPoolOperator.h"

Tensor softmax(const Tensor& t) {
	auto out = t - (*t).maxCoeff();
	Tensor tmp(t.col(), t.col());
	tmp.setOnes();
	return out.exp().dot((out.exp() * tmp).pow(-1));
}

Tensor maxPool(const Tensor& t, int channel, int dataRow, int dataCol, int kernelRow, int kernelCol, int stride) {
	return (*shared_ptr<Operator>(new MaxPoolOperator(t, channel, dataRow, dataCol, kernelRow, kernelCol, stride)))();
}