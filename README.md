
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
