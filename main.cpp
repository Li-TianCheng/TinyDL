#include <iostream>
#include "Tensor.h"
#include "model/Model.h"
#include "model/Linear.h"
#include "model/BatchNorm.h"
#include "model/ActivateFun.h"
#include "model/LossFun.h"
#include "optimizer/AdamOptimizer.h"

using namespace std;

class Example : public Model {
public:
	Tensor forward(const Tensor &input) override {
		auto out = fc1(input);
		out = b1(out);
		out = relu(out);
		out = fc2(out);
		out = sigmoid(out);
		out = fc3(out);
		out = tanh(out);
		out = fc4(out);
		return out;
	}
private:
	Linear fc1 = Linear(*this, 10, 400);
	BatchNorm b1 = BatchNorm(*this, 400);
	Linear fc2 = Linear(*this, 400, 200);
	Linear fc3 = Linear(*this, 200, 100);
	Linear fc4 = Linear(*this, 100, 10);
};

int main() {
	// TODO：Cnn、BatchNorm、Rnn、Lstm
	Tensor t(100, 10);
	t.setRandom();
	Example e;
	AdamOptimizer op(e.parameters);
	Tensor label(100,10);
	label.setRandom();

	for (int i = 0; i < 100; i++) {
		op.clearGradient();
		Tensor out = e(t);
		Tensor loss = MSELoss(out, label);
		cout << "epoch: " << i+1 << " loss: " << loss << endl;
		loss.backward();
		op.step();
	}
    return 0;
}
