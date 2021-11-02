#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include "Tensor.h"
#include "model/Model.h"
#include "model/Linear.h"
#include "model/Convolution.h"
#include "model/ActivateFun.h"
#include "model/LossFun.h"
#include "model/UtilsFun.h"
#include "optimizer/AdamOptimizer.h"

using namespace std;

class Example : public Model {
public:
	Tensor forward(const Tensor &input) override {
		auto out = conv1(input);
		out = relu(out);
		out = maxPool(out, 20, 24, 24, 2, 2, 2);
		out = conv2(out);
		out = relu(out);
		out = maxPool(out, 50, 8, 8, 2, 2, 2);
		out = fc1(out);
		out = relu(out);
		out = fc2(out);
		return out;
	}
private:
	Convolution conv1 = Convolution(*this, 1, 20, 28, 28, 5, 5, 1);
	Convolution conv2 = Convolution(*this, 20, 50, 12, 12, 5, 5, 1);
	Linear fc1 = Linear(*this, 50*4*4, 1024);
	Linear fc2 = Linear(*this, 1024, 10);
};

int reverseInt(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

vector<vector<double>> readMnistData(const string& filename) {
	vector<vector<double>> data;
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char *) &magic_number, sizeof(magic_number));
		file.read((char *) &number_of_images, sizeof(number_of_images));
		file.read((char *) &n_rows, sizeof(n_rows));
		file.read((char *) &n_cols, sizeof(n_cols));
		magic_number = reverseInt(magic_number);
		number_of_images = reverseInt(number_of_images);
		n_rows = reverseInt(n_rows);
		n_cols = reverseInt(n_cols);
		for (int i = 0; i < number_of_images; i++) {
			vector<double> tp;
			for (int r = 0; r < n_rows; r++) {
				for (int c = 0; c < n_cols; c++) {
					unsigned char image = 0;
					file.read((char *) &image, sizeof(image));
					tp.push_back((double)image/255);
				}
			}
			data.push_back(tp);
		}
	}
	return data;
}

vector<double> readMnistLabel(const string& filename) {
	vector<double> labels;
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char *) &magic_number, sizeof(magic_number));
		file.read((char *) &number_of_images, sizeof(number_of_images));
		magic_number = reverseInt(magic_number);
		number_of_images = reverseInt(number_of_images);
		for (int i = 0; i < number_of_images; i++) {
			unsigned char label = 0;
			file.read((char *) &label, sizeof(label));
			labels.push_back((double) label);
		}
	}
	return labels;
}

int main() {
	vector<vector<double>> mnistData = readMnistData("../data/train-images.idx3-ubyte");
	vector<double> mnistLabel = readMnistLabel("../data/train-labels.idx1-ubyte");
	Example e;
	e.cuda();
	AdamOptimizer op(e.parameters);
	int batchSize = 128;
	int index = 0;
	vector<int> idx(mnistData.size());
	for (int i = 0; i < idx.size(); ++i) {
		idx[i] = i;
	}
	std::shuffle(idx.begin(), idx.end(), std::mt19937(std::random_device()()));
	vector<vector<double>> batchData(batchSize);
	vector<double> batchLabel(batchSize);
	for (int i = 0; i < 100; ++i) {
		for (int j = 0; j < batchSize; ++j) {
			batchData[j] = mnistData[idx[index]];
			batchLabel[j] = mnistLabel[idx[index]];
			index = (index+1) % (int)mnistData.size();
		}
		Tensor data(batchData, true);
		Tensor label({batchLabel}, true);
		op.clearGradient();
		Tensor out = e(data);
		Tensor loss = crossEntropyLoss(out, label);
		int rightNum = 0;
		for (int m = 0; m < out.row(); ++m) {
			int maxIdx = 0;
			for (int n = 0; n < out.col(); ++n) {
				if (out(m, n) > out(m, maxIdx)) {
					maxIdx = n;
				}
			}
			if (maxIdx == label(0, m)) {
				rightNum++;
			}
		}
		cout << "step: " << i+1 << " loss: " << loss(0, 0) << " acc: " << (double)rightNum/batchSize << endl;
		loss.backward();
		op.step();
	}
    return 0;
}
