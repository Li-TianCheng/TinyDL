//
// Created by ltc on 2021/9/9.
//

#ifndef TINYDL_MODEL_H
#define TINYDL_MODEL_H

#include <vector>
#include "Tensor.h"

using std::vector;

class Model {
public:
	Tensor operator()(const Tensor& input);
	virtual Tensor forward(const Tensor& input) = 0;
	virtual ~Model() = default;
public:
	vector<Tensor> parameters;
};


#endif //TINYDL_MODEL_H
