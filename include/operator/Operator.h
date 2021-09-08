//
// Created by ltc on 2021/9/2.
//

#ifndef TINYDL_OPERATOR_H
#define TINYDL_OPERATOR_H

#include <cmath>
#include "Tensor.h"

class Operator : public std::enable_shared_from_this<Operator> {
public:
	Operator(const Tensor& tensor1, const Tensor& tensor2);
	Operator(const Operator& op) = delete;
	Operator(Operator&& op) = delete;
	Operator& operator=(const Operator& op) = delete;
	Operator& operator=(Operator&& op) = delete;
	virtual Tensor operator()() = 0;
	virtual void grad(Tensor& result) = 0;
protected:
	friend class Tensor;
	Tensor tensor1;
	Tensor tensor2;
};


#endif //TINYDL_OPERATOR_H
