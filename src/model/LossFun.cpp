//
// Created by ltc on 2021/9/10.
//

#include "model/LossFun.h"

Tensor MSELoss(Tensor& pred, Tensor& label) {
	auto out = pred;
	pred.freeOperator();
	Tensor t1(1, pred.row(), pred.isCuda());
	Tensor t2(pred.col(), 1, pred.isCuda());
	t1.setOnes();
	t2.setOnes();
	return t1 * (out - label).pow(2) * t2 / pred.row();
}

Tensor crossEntropyLoss(Tensor& pred, Tensor& label) {
	auto out = softmax(pred) + 1e-9;
	pred.freeOperator();
	Tensor l(pred.row(), pred.col(), pred.isCuda());
	for (int i = 0; i < label.col(); ++i) {
		(*l).setValue(i, (int)label(0, i), -1);
	}
	Tensor t1(1, pred.row(), pred.isCuda());
	Tensor t2(pred.col(), 1, pred.isCuda());
	t1.setOnes();
	t2.setOnes();
	return t1 * out.log().dot(l) * t2 / pred.row();
}