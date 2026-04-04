# thesis-safe-activation

## 课题名称
用于模型非线性函数替代的安全算法模块设计

## 研究目标
设计一个面向 HE/MPC 场景的非线性函数替代模块，支持 ReLU、Sigmoid、GELU、Softmax 的多项式近似，
能够以可插拔方式嵌入 PyTorch 模型，并通过小型 CNN 在 MNIST 上验证其对模型精度、收敛情况和计算效率的影响。

## 当前项目结构
- safe_activations/：原函数、近似函数、模块封装、自动替换
- models/：模型定义
- experiments/：训练与评估脚本
- plots/：可视化脚本
- tests/：测试代码
- outputs/：图表、日志、实验结果
- thesis/：论文草稿

## 计划中的主要工作
1. 实现 ReLU、Sigmoid、GELU、Softmax 的多项式近似
2. 封装统一接口 f(x) / approx_f(x)
3. 实现 PyTorch 激活层替换模块
4. 在 MNIST 小型 CNN 上完成对照实验
5. 输出函数图像、误差分析和模型性能对比结果

## 当前进度
- [x] 项目初始化
- [x] 函数近似模块
- [x] 自动替换机制
- [ ] MNIST 模型实验
- [ ] 消融实验
- [ ] 论文撰写