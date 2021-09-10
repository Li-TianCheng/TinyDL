//
// Created by ltc on 2021/9/10.
//

#include "model/LossFun.h"

Tensor MSELoss(const Tensor& pred, const Tensor& label) {
	Tensor t1(1, pred.row());
	Tensor t2(pred.col(), 1);
	t1.setOnes();
	t2.setOnes();
	return t1 * (pred - label).pow(2) * t2 / pred.row();
}

Tensor crossEntropyLoss(const Tensor& pred, const Tensor& label) {
	Tensor t1(1, pred.row());
	Tensor t2(pred.col(), 1);
	t1.setOnes();
	t2.setOnes();
	return t1 * (pred.dot(label)).log() * t2 / -pred.row();
}