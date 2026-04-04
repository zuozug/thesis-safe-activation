| 编号   | 实验                | 脚本                                                     | 作用                     | 是否进论文主体  |
| ---- | ----------------- | ------------------------------------------------------ | ---------------------- | -------- |
| 实验 0 | 联调与环境检查           | `tests/test_integration.py` + `test_day2/3/4.py`       | 先确认代码链路全通，避免后面训练白跑     | 作为实验准备说明 |
| 实验 1 | Baseline CNN      | `experiments/train_baseline.py`                        | 建立对照基线                 | 是        |
| 实验 2 | ApproxReLU CNN    | `experiments/train_approx_relu.py`                     | 验证 ReLU 近似替代效果         | 是，主实验    |
| 实验 3 | ApproxGELU CNN    | `experiments/train_approx_gelu.py`                     | 验证 GELU 近似替代效果         | 是，主实验    |
| 实验 4 | ApproxSigmoid CNN | `experiments/train_approx_sigmoid.py`                  | 补充平滑激活函数实验             | 是，补充实验   |
| 实验 5 | Softmax 输出层评估     | `experiments/eval_softmax.py`                          | 验证 Softmax 近似的分布误差与稳定性 | 是，补充实验   |
| 实验 6 | ReLU 消融           | `experiments/run_ablation.py --hidden-activation relu` | 比较不同阶数、不同区间            | 是        |
| 实验 7 | GELU 消融           | `experiments/run_ablation.py --hidden-activation gelu` | 比较不同阶数、不同区间            | 是        |

## 实验 0：联调与环境检查

### 实验目的
在正式开展模型训练实验前，对项目的整体代码链路进行联调检查，验证近似函数模块、模块封装、自动替换机制以及测试框架是否能够正常运行，避免后续 Baseline CNN 和近似模型训练实验因环境或接口问题白跑。

本实验对应实验表中的“实验 0：联调与环境检查”，其作用是先确认代码链路全通，作为后续正式实验的准备步骤。

### 运行命令
本次通过 PyCharm 内置 unittest 运行器执行：

```bash
D:\code\thesis-safe-activation\.venv\Scripts\python.exe C:/Users/27454/AppData/Local/Programs/PyCharm/plugins/python-ce/helpers/pycharm/_jb_unittest_runner.py --path D:\code\thesis-safe-activation\tests\test_integration.py
```
PyCharm 实际调用的 unittest 命令为：

```bash
python -m unittest D:\code\thesis-safe-activation\tests\test_integration.py --quiet
```

### 配置

* 操作系统：Windows
* IDE：PyCharm
* Python 环境：`.venv`
* 测试脚本：`tests/test_integration.py`
* 工作目录：`D:\code\thesis-safe-activation`
* Python 版本：3.14.3
* PyTorch 版本：2.11.0
* torchvision 版本：0.26.0

### 结果

测试输出如下：

```text
Ran 23 tests in 1.055s

OK

进程已结束，退出代码为 0
```

结果表明：

* 测试脚本成功启动；
* 共执行 23 项测试；
* 所有测试均通过；
* 未出现 FAIL 或 ERROR；
* 进程正常结束。

### 分析

实验 0 的目标不是获得模型精度结果，而是验证项目当前代码是否具备进入正式训练阶段的条件。
本次 `test_integration.py` 的 23 项测试全部通过，说明当前项目的核心模块已经能够在现有环境下正常工作，至少不存在明显的导入错误、接口不匹配或基础链路中断问题。

这意味着：

1. 当前 Python 虚拟环境可正常运行项目测试；
2. 项目核心代码结构已基本稳定；
3. 后续可以继续开展 Baseline CNN、ApproxReLU CNN、ApproxGELU CNN 等正式训练实验；
4. 实验 0 达到了“避免后续训练白跑”的预期目的。

需要注意的是，本实验属于“实验准备说明”，不直接产出准确率、损失值、训练时间等模型性能指标，因此在论文正文中只需简要说明其作用，不必展开成主体结果表。

### 结论

实验 0 通过。
项目当前已完成基础联调与环境检查，可以进入下一步正式实验，即实验 1：Baseline CNN。

## 实验 1：Baseline CNN

### 实验目的
使用原始激活函数的基线 CNN 在 MNIST 数据集上进行训练与测试，建立后续 ApproxReLU、ApproxGELU 和 ApproxSigmoid 模型的对照基线。重点记录验证准确率、测试准确率、损失、训练时间和推理时间，为后续近似激活函数模型的性能比较提供参考。

### 运行命令
```bash
py -m experiments.train_baseline --device cpu --hidden-activation relu --batch-size 128 --epochs 8 --lr 1e-3 --val-ratio 0.1 --num-workers 0 --seed 42 --output-dir outputs/logs/exp1_baseline
````

### 配置

* 数据集：MNIST
* 模型：Baseline CNN
* 隐藏层激活函数：ReLU
* batch size：128
* epochs：8
* learning rate：1e-3
* val ratio：0.1
* num_workers：0
* seed：42
* device：cpu

### 结果

* best val accuracy：0.9888333333333333
* final val loss：0.04454690021773179
* final val accuracy：0.9888333333333333
* final test loss：0.03660337060673046
* final test accuracy：0.9884
* total training seconds：70.65932109998539
* average epoch seconds：8.827397374960128
* inference seconds per batch：0.003248864982742816
* inference seconds per sample：0.00002538175767767825

### 分析

本实验中，Baseline CNN 在 MNIST 数据集上表现出稳定的收敛趋势。随着训练轮数增加，训练损失由 0.3436 持续下降至 0.0197，验证损失由 0.1208 下降至 0.0445，验证准确率由 0.9648 提升至 0.9888，说明模型训练过程平稳且有效。

最终测试准确率达到 0.9884，与最佳验证准确率 0.9888 非常接近，表明模型具有较好的泛化能力，没有出现明显过拟合现象。该结果说明当前基线模型已经能够在 MNIST 上取得较高分类性能，可以作为后续近似激活函数模型实验的可靠对照基线。

在效率方面，总训练时间约为 70.66 秒，平均每轮训练约 8.83 秒；推理阶段平均每个 batch 用时约 0.00325 秒，每个样本平均推理时间约 2.54e-05 秒。这些指标将作为后续 ApproxReLU、ApproxGELU 和 ApproxSigmoid 实验的时间比较基准。

### 结论

实验 1 顺利完成，Baseline CNN 基线已建立。该模型在当前设置下取得了较高的测试准确率和较稳定的训练表现，能够作为后续近似激活函数模型对照实验的基准模型。

## 实验 2：ApproxReLU CNN

### 实验目的
在 MNIST 数据集上训练采用 ReLU 多项式近似替代的 CNN 模型，验证 ApproxReLU 对模型精度、收敛情况、训练时间和推理时间的影响，并与 Baseline CNN 进行对照分析。

### 运行命令
```bash
py -m experiments.train_approx_relu --device cpu --batch-size 128 --epochs 8 --lr 1e-3 --val-ratio 0.1 --num-workers 0 --seed 42 --degree 4 --interval-left -3 --interval-right 3 --method chebyshev --output-dir outputs/logs/exp2_approx_relu
````

### 配置

* 数据集：MNIST
* 模型：ApproxReLU CNN
* 近似目标函数：ReLU
* 近似方法：Chebyshev
* 多项式阶数：4
* 逼近区间：[-3, 3]
* batch size：128
* epochs：8
* learning rate：1e-3
* val ratio：0.1
* num_workers：0
* seed：42
* device：cpu

### 结果

* best val accuracy：0.9888333333333333
* final val loss：0.03839651202162107
* final val accuracy：0.9888333333333333
* final test loss：0.030585972418589517
* final test accuracy：0.9903
* total training seconds：118.1076330000069
* average epoch seconds：14.761183562513907
* inference seconds per batch：0.006234150019008666
* inference seconds per sample：0.000048704297023505204

### 分析

ApproxReLU 模型在本次实验中表现出稳定的收敛趋势。随着训练轮数增加，训练损失由 0.4274 持续下降至 0.0142，验证损失由 0.1117 下降至 0.0384，验证准确率最高达到 0.9888，说明模型训练过程平稳有效。

与 Baseline CNN 对比可知，ApproxReLU 的最佳验证准确率与基线模型完全一致，测试准确率由 0.9884 提升至 0.9903，表明在当前 4 阶、区间 [-3, 3]、Chebyshev 近似配置下，ReLU 多项式替代并未造成模型精度下降，反而在本次实验中取得了略优的测试表现。同时，ApproxReLU 的验证损失和测试损失均低于 Baseline，说明该近似替代在保持分类性能方面具有较好的可行性。

但在效率方面，ApproxReLU 的训练和推理时间明显高于 Baseline：总训练时间由 70.66 秒增加至 118.11 秒，平均每轮训练时间由 8.83 秒增加至 14.76 秒，推理每 batch 时间由 0.00325 秒增加至 0.00623 秒。这说明在普通明文 CPU 环境下，多项式近似替代并不会直接带来速度优势，反而会引入一定的时间开销。

综合来看，ApproxReLU 的主要价值不在于当前明文环境下的加速，而在于将原本含比较操作的 ReLU 转化为仅含加法和乘法的多项式表达形式，从而为未来接入 HE/MPC 等隐私计算场景提供结构上的兼容性基础。

### 结论

实验 2 顺利完成。结果表明，在 4 阶、区间 [-3, 3]、Chebyshev 配置下，ApproxReLU 能够在 MNIST 上保持与 Baseline CNN 相当的验证精度，并取得略高的测试准确率，说明 ReLU 多项式近似替代方案具有良好的可行性；但其代价是在普通 CPU 环境下训练与推理时间明显增加。

## 实验 3：ApproxGELU CNN

### 实验目的
在 MNIST 数据集上训练采用 GELU 多项式近似替代的 CNN 模型，验证 ApproxGELU 对模型精度、收敛情况、训练时间和推理时间的影响，并与 Baseline CNN 及 ApproxReLU CNN 进行对照分析。

### 运行命令
```bash
py -m experiments.train_approx_gelu --device cpu --batch-size 128 --epochs 8 --lr 1e-3 --val-ratio 0.1 --num-workers 0 --seed 42 --degree 5 --interval-left -3 --interval-right 3 --method chebyshev --output-dir outputs/logs/exp3_approx_gelu
````

### 配置

* 数据集：MNIST
* 模型：ApproxGELU CNN
* 近似目标函数：GELU
* 近似方法：Chebyshev
* 多项式阶数：5
* 逼近区间：[-3, 3]
* batch size：128
* epochs：8
* learning rate：1e-3
* val ratio：0.1
* num_workers：0
* seed：42
* device：cpu

### 结果

* best val accuracy：0.9898333333333333
* final val loss：0.03612672908604145
* final val accuracy：0.9898333333333333
* final test loss：0.028730415977025404
* final test accuracy：0.9901
* total training seconds：111.91607449995354
* average epoch seconds：13.987715099967318
* inference seconds per batch：0.005924535018857568
* inference seconds per sample：0.00004628542983482475

### 分析

ApproxGELU 模型在本次实验中表现出稳定的收敛趋势。随着训练轮数增加，训练损失由 0.3328 持续下降至 0.0114，验证损失由 0.1037 下降至 0.0361，验证准确率最高达到 0.9898，说明模型训练过程平稳有效。

与 Baseline CNN 对比可知，ApproxGELU 的最佳验证准确率由 0.9888 提升至 0.9898，测试准确率由 0.9884 提升至 0.9901；同时，其验证损失和测试损失均低于基线模型。这说明在当前 5 阶、区间 [-3, 3]、Chebyshev 近似配置下，GELU 多项式替代并未导致模型性能下降，反而取得了更低损失和略高精度。

与 ApproxReLU 对比，ApproxGELU 的测试准确率略低 0.0002，但验证准确率更高，验证损失和测试损失也更低，说明其整体表现更加均衡。可以认为，ApproxGELU 和 ApproxReLU 在当前实验中的测试精度接近，但 ApproxGELU 在验证集表现和损失指标上更有优势。

在效率方面，ApproxGELU 的训练和推理时间仍明显高于 Baseline，说明在普通明文 CPU 环境下，多项式近似替代不会直接带来速度提升；但与 ApproxReLU 相比，ApproxGELU 的总训练时间和推理时间略低，表明其在当前配置下具有更好的综合效率表现。

综合来看，ApproxGELU 的主要价值在于将原本复杂的 GELU 激活转化为更适合 HE/MPC 场景的多项式形式，同时在明文实验中保持了较高精度和较低损失，具备较好的工程可行性。

### 结论

实验 3 顺利完成。结果表明，在 5 阶、区间 [-3, 3]、Chebyshev 配置下，ApproxGELU 能够在 MNIST 上保持优良的分类性能，并取得比 Baseline 更高的验证与测试精度；同时，其整体表现与 ApproxReLU 非常接近，且在损失和部分时间指标上更均衡，说明 GELU 多项式近似替代方案具有良好的可行性。
