//
// Created by ltc on 2021/9/13.
//

#include "model/Utils.h"

Tensor softmax(const Tensor& t) {
	Tensor tmp(t.col(), t.col());
	tmp.setOnes();
	return t.exp().dot((t.exp() * tmp).pow(-1));
}

Tensor maxPool(const Tensor& t, int kernelRow, int kernelCol, int stride) {

}