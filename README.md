# TinyDL

## 基于Eigen运算库的深度学习框架(支持CUDA加速)

### 安装
1. Eigen
2. MKL
3. CUDA

### 使用
* Tensor默认是分配在cpu上,可在申请Tensor时设置cuda为true将Tensor分配在cuda上
* Tensor可以使用cuda()或cpu()将Tensor迁移到cuda或cpu
* Model和Optimizer可使用cuda()或cpu()迁移到cuda或cpu

### 实现模块

#### 网络模块
1. 线性层
2. 卷积层
3. Batch normal

#### 损失函数
1. 均方误差(MSE loss)
2. 交叉熵(cross entropy loss)

#### 激活函数
1. sigmoid
2. tanh
3. relu

#### 优化器
1. SGD
2. Adam

#### 其他常用函数
1. softmax
2. max pool