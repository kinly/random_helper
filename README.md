
# inlay::random

一个高性能、线程安全的 C++ 随机数工具库，支持线程局部随机引擎、通用区间采样函数 `range`，以及三种不同策略的权重采样器（Alias、Expansion、Binary）。

## 特性 Features

- 💡 线程局部随机引擎封装：`thread_rng()`
- 🎲 通用区间采样函数：`range(min, max)`
- 📦 三种权重采样器：
  - `weight_faster_alias`: O(1) 采样，适合频繁调用
  - `weight_faster_expansion`: 极致快速采样，适合小权重集合
  - `weight_faster_binary`: 适合通用场景，支持整数/浮点权重

## 示例 Example

```cpp
#include "random.hpp"
#include <iostream>

int main() {
  std::vector<int> values{1, 2, 3, 4};
  std::vector<double> weights{0.1, 0.2, 0.3, 0.4};

  inlay::random::weight_faster_alias<int> sampler(
      values.begin(), values.end(),
      weights.begin(), weights.end()
  );

  for (int i = 0; i < 10; ++i) {
    std::cout << sampler() << " ";
  }
}
```

## 性能比较 Benchmark

| 方法                  | 初始化复杂度 | 采样复杂度 | 适用场景说明 |
|-----------------------|---------------|-------------|----------------|
| `Alias`               | O(n)          | O(1)        | 高频采样、浮点权重 |
| `Expansion`           | O(n × w)      | O(1)        | 小权重集合、最快采样 |
| `Binary`              | O(n)          | O(log n)    | 通用、整数权重优先 |

## 编译要求

- C++20 及以上（仅少量语法）
- 无第三方依赖（仅 STL）

## TODO

- [ ] 添加更多采样策略支持
- [ ] 支持分布策略插件式扩展


## 信息备忘

>> 引擎（Engine）：像是一台骰子机，不停地吐出“基础随机数”。

>> 分布（Distribution）：是一个数学过滤器，告诉你“这些随机数应该分布成什么样子”。

- 如何选择随机引擎

|应用场景 | 推荐引擎 | 原因|
|----------|----------|----------|
|游戏中的非关键随机行为 | default_random_engine 或 minstd_rand | 快速，易于使用|
|科学模拟 | mt19937 或 ranlux48 | 高质量随机数|
|加密相关 | ❌ 不使用这些引擎 | 请使用 std::random_device + 专业加密库|
|教学 / 学习 | minstd_rand, knuth_b | 简单，容易理解|
|嵌入式系统 | minstd_rand0 | 小内存，占用低|
|跨平台一致性要求 | 明确指定 mt19937 并使用固定种子 | 保证可重现性|

- 常见分布种类及用途

|分布名称 | 类名 | 描述 | 典型用途|
|----------|----------|----------|----------|
|均匀整数分布 | std::uniform_int_distribution | 等概率生成某范围内的整数 | 抽奖、骰子、游戏事件|
|均匀实数分布 | std::uniform_real_distribution | 生成实数范围内等概率值 | 物理模拟、归一化值|
|伯努利分布 | std::bernoulli_distribution | 成功概率 p 的二项分布（0 或 1） | 掷硬币、概率判断|
|二项分布 | std::binomial_distribution | n 次试验中成功次数 | 离散事件成功次数模拟|
|几何分布 | std::geometric_distribution | 成功一次前失败的次数 | 等待第一次成功的模型|
|泊松分布 | std::poisson_distribution | 单位时间内事件发生次数 | 网络包、电话中心模拟|
|正态分布（高斯） | std::normal_distribution | 钟形曲线 | 噪声、误差模拟、机器学习初始化|
|对数正态分布 | std::lognormal_distribution | 变量对数正态分布 | 股票收益、金融模型|
|指数分布 | std::exponential_distribution | 时间间隔模拟 | 排队、寿命建模|
|gamma 分布 | std::gamma_distribution | 广义泊松时间间隔 | 生物模型、贝叶斯方法|
|Weibull 分布 | std::weibull_distribution | 产品寿命分析 | 工业可靠性|
|extreme value 分布 | std::extreme_value_distribution | 极端事件概率 | 风速、洪水预测|