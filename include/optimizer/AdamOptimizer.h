//
// Created by ltc on 2021/9/10.
//

#ifndef TINYDL_ADAMOPTIMIZER_H
#define TINYDL_ADAMOPTIMIZER_H

#include "optimizer/Optimizer.h"

class AdamOptimizer : public Optimizer {
public:
	explicit AdamOptimizer(const vector<Tensor>& parameters, double lr=1e-3, double beta1=0.9, double beta2=0.999, double eps=1e-8);
	void step() override;
private:
	int t;
	double lr;
	double eps;
	double beta1;
	double beta2;
	vector<Tensor> m;
	vector<Tensor> v;
};


#endif //TINYDL_ADAMOPTIMIZER_H
