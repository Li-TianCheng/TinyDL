//
// Created by ltc on 2021/9/10.
//

#ifndef TINYDL_OPTIMIZER_H
#define TINYDL_OPTIMIZER_H

#include <vector>
#include "Tensor.h"

using std::vector;

class Optimizer {
public:
	explicit Optimizer(const vector<Tensor>& parameters);
	void clearGradient();
	virtual void step() = 0;
	virtual ~Optimizer() = default;
protected:
	vector<Tensor> parameters;
};


#endif //TINYDL_OPTIMIZER_H
