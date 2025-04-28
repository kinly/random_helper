#pragma once
#include <cassert>
#include <numeric>
#include <queue>
#include <random>

namespace inlay::random {

using engine = std::default_random_engine;

inline engine& thread_rng(std::optional<uint32_t> seed = std::nullopt) {
  thread_local std::optional<engine> eng;
  if (seed.has_value()) {
    eng.emplace(seed.value());
  } else if (!eng.has_value()) {
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    eng.emplace(static_cast<uint32_t>(now ^ tid));
  }
  return *eng;
}

template <class tt>
  requires std::integral<tt> || std::floating_point<tt>
tt range(tt min_value, tt max_value, engine& rng = thread_rng()) {
  if (min_value > max_value)
    std::swap(min_value, max_value);

  if constexpr (std::integral<tt>)
    return std::uniform_int_distribution<tt>(min_value, max_value)(rng);
  else
    return std::uniform_real_distribution<tt>(min_value, max_value)(rng);
}

/* Walker's Alias Method 代码内部做了权重修正 */
// https://www.keithschwarz.com/darts-dice-coins/
//
// 时间复杂度:
//   初始化：O(n)（构建 Alias 表）
//   采样操作：O(1)
// 空间复杂度:
//   O(n)
// 适用:
//   适合频繁采样（采样操作快）
//
template <class tt>
class weight_faster_alias {
 private:
  using value_type = tt;
  const std::vector<value_type> _values;
  std::vector<std::pair<double, size_t>> _alias;
  mutable std::uniform_real_distribution<double> _real_dis{0.0, 1.0};
  mutable std::uniform_int_distribution<size_t> _int_dis;
  engine& _rng;
 public:
  template <typename v_tt, typename p_tt>
  weight_faster_alias(
    const v_tt vs, const v_tt ve, const p_tt ps, const p_tt pe, engine& rng = thread_rng())
      : _values(vs, ve), _int_dis(0, std::distance(ps, pe) - 1), _rng(rng) {

    // assert(std::distance(vs, ve) != 0 && std::distance(vs, ve) == std::distance(ps, pe));
    if (std::distance(vs, ve) == 0 || std::distance(vs, ve) != std::distance(ps, pe)) {
      throw std::invalid_argument("value range and weight range must be non-empty and equal size");
    }

    std::vector<double> reprods{};
    const double sum_prod = std::accumulate(ps, pe, 0.0);
    if (std::fabs(1.0 - sum_prod) > std::numeric_limits<double>::epsilon()) {
      for (auto iter = ps; iter != pe; ++iter) {
        reprods.push_back(*iter / sum_prod);
      }
    } else {
      reprods.insert(reprods.end(), ps, pe);
    }
    const double re_sum_prod = std::accumulate(reprods.begin(), reprods.end(), 0.0);
    // assert(std::fabs(1.0 - re_sum_prod) < std::numeric_limits<double>::epsilon());
    if (std::fabs(1.0 - re_sum_prod) >= std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error("Sum of normalized weights is not 1.0");
    }
    _alias = generate_alias_table(reprods.begin(), reprods.end());
  }

  value_type operator()() const noexcept {
    const size_t idx = _int_dis(_rng);
    if (_real_dis(_rng) >= _alias[idx].first && _alias[idx].second != std::numeric_limits<size_t>::max()) {
      return _values[_alias[idx].second];
    } else {
      return _values[idx];
    }
  }

 private:
  template <typename rp_tt>
  std::vector<std::pair<double, size_t>> generate_alias_table(const rp_tt ps, const rp_tt pe) {
    const size_t sz = std::distance(ps, pe);
    std::vector<std::pair<double, size_t>> alias(sz, {0.0, std::numeric_limits<size_t>::max()});
    std::queue<size_t> small, large;

    auto iter = ps;
    size_t i = 0;
    for (; iter != pe; ++iter, ++i) {
      alias[i].first = sz * (*iter);
      if (alias[i].first < 1.0) {
        small.push(i);
      } else {
        large.push(i);
      }
    }

    while (!(small.empty()) and !(large.empty())) {
      auto s = small.front(), l = large.front();
      small.pop(), large.pop();
      alias[s].second = l;
      alias[l].first -= (1.0 - alias[s].first);

      if (alias[l].first < 1.0) {
        small.push(l);
      } else {
        large.push(l);
      }
    }

    return alias;
  }
};

/* Expansion 适合总和较小的整数权重，空间换时间，这个最快 */
//
// 时间复杂度:
//   初始化：O(n* w)（将每个元素展开成 w 次，w 为该元素权重）
//   采样操作：O(1)（直接 range(0, size - 1)）
// 空间复杂度:
//   O(W)，其中 W = 所有权重之和
// 适用:
//   权重总和 较小，如权重都是个位数
//   对采样速度有极致追求，最快采样性能
//   不适合大规模、高权重的情况（内存爆炸）
//   相比 Alias 的浮点数，采样稍快一点点
//
template <class tt, class prods_tt = uint32_t>
class weight_faster_expansion {
 private:
  using value_type = tt;
  using prod_type = prods_tt;
  std::vector<value_type> _values;
  engine& _rng;

 public:
  template <typename v_tt, typename p_tt>
  weight_faster_expansion(const v_tt vs, const v_tt ve, const p_tt ps, const p_tt pe, engine& rng = thread_rng())
      : _rng(rng) {
    // assert(std::distance(vs, ve) != 0 && std::distance(vs, ve) == std::distance(ps, pe));
    if (std::distance(vs, ve) == 0 || std::distance(vs, ve) != std::distance(ps, pe)) {
      throw std::invalid_argument("value range and weight range must be non-empty and equal size");
    }
    const prod_type sum_prod = std::accumulate(ps, pe, 0);
    auto iter_v = vs;
    auto iter_p = ps;
    for (; iter_v != ve && iter_p != pe; ++iter_v, ++iter_p) {
      _values.insert(_values.end(), *iter_p, *iter_v);
    }
  }

  value_type operator()() noexcept {
    const size_t idx = range(static_cast<size_t>(0), _values.size() - 1, _rng);
    return _values[idx];
  }
};

/* binary 一般性权重随机 */
//
// 时间复杂度:
//   初始化：O(n)（构建前缀和表）
//   采样操作：O(log n)（使用 std::lower_bound 二分查找）
// 空间复杂度: O(n)：
//   存储原始值 + 前缀和表
// 适用:
//   元素数量不是特别多（log n 能接受）
//   初始化和采样速度都适中
//   权重是整数或浮点均可（推荐整数）
//
template <class tt, class prods_tt = uint32_t>
class weight_faster_binary {
 private:
  using value_type = tt;
  using prod_type = prods_tt;
  std::vector<value_type> _values;
  std::vector<prod_type> _prods;
  engine& _rng;

 public:
  template <typename v_tt, typename p_tt>
  weight_faster_binary(const v_tt vs, const v_tt ve, const p_tt ps, const p_tt pe, engine& rng = thread_rng())
    : _rng(rng) {
    // assert(std::distance(vs, ve) != 0 && std::distance(vs, ve) == std::distance(ps, pe));
    if (std::distance(vs, ve) == 0 || std::distance(vs, ve) != std::distance(ps, pe)) {
      throw std::invalid_argument("value range and weight range must be non-empty and equal size");
    }

    _values.insert(_values.end(), vs, ve);

    prod_type last_sum = 0;
    for (auto iter = ps; iter != pe; ++iter) {
      last_sum += *iter;
      _prods.push_back(last_sum);
    }
  }

  value_type operator()() noexcept {
    const auto rand = range(static_cast<prod_type>(1), _prods.back(), _rng);
    auto idx = std::lower_bound(_prods.begin(), _prods.end(), rand) - _prods.begin();
    return _values[idx];
  }
};

};  // end namespace inlay::random


// template <class _Period = std::micro>
// class timer_cost {
//   std::chrono::steady_clock::time_point _last_time_point;
//
//  public:
//   timer_cost() : _last_time_point(std::chrono::steady_clock::now()) {}
//
//   void print(const std::string& msg) {
//     auto now = std::chrono::steady_clock::now();
//     std::chrono::duration<double, _Period> diff{now - _last_time_point};
//     std::cout << msg << " " << diff.count() << std::endl;
//     _last_time_point = std::chrono::steady_clock::now();
//   }
// };

// void test_random_helper(bool same_seed = false) {
//   std::vector<int> v{1, 2, 3, 4, 5, 6, 7, 8, 9};
//   std::vector<uint32_t> p{1, 2, 3, 4, 5, 6, 7, 8, 9};
//   std::vector<double> dp{1 / 55.0, 2 / 55.0, 3 / 55.0, 4 / 55.0, 5 / 55.0, 6 / 55.0, 7 / 55.0, 8 / 55.0, 9 / 55.0};
//
//   // must copy result engine
//   auto wfa_engine = same_seed ? inlay::random::thread_rng(1) : inlay::random::thread_rng();
//   auto wfa_2_engine = same_seed ? inlay::random::thread_rng(1) : inlay::random::thread_rng();
//   auto wfs_engine = same_seed ? inlay::random::thread_rng(1) : inlay::random::thread_rng();
//   auto wfb_engine = same_seed ? inlay::random::thread_rng(1) : inlay::random::thread_rng();
//
//   inlay::random::weight_faster_alias<int> wfa(v.begin(), v.end(), dp.begin(), dp.end(), wfa_engine);
//   inlay::random::weight_faster_alias<int> wfa_2(v.begin(), v.end(), p.begin(), p.end(), wfa_2_engine);
//   inlay::random::weight_faster_expansion<int> wfs(v.begin(), v.end(), p.begin(), p.end(), wfs_engine);
//   inlay::random::weight_faster_binary<int> wfb(v.begin(), v.end(), p.begin(), p.end(), wfb_engine);
//
//   if (true) {
//     auto& engine_1 = inlay::random::thread_rng(1);
//     for (int i = 0; i < 10; ++i) {
//       std::cout << inlay::random::range(1, 10, engine_1) << '\t';
//     }
//     std::cout << '\n';
//     auto& engine_2 = inlay::random::thread_rng(1);
//     for (int i = 0; i < 10; ++i) {
//       std::cout << inlay::random::range(1, 10, engine_2) << '\t';
//     }
//   }
//
//   std::map<int, int> wfa_statistics;
//   std::map<int, int> wfa_2_statistics;
//   std::map<int, int> wfs_statistics;
//   std::map<int, int> wfb_statistics;
//
//   timer_cost tc;
//
//   for (int i = 0; i < 40000; ++i) {
//     auto wfa_index = wfa();
//     wfa_statistics[wfa_index] += 1;
//   }
//   tc.print("wfa:");
//
//   for (int i = 0; i < 40000; ++i) {
//     auto wfa_2_index = wfa_2();
//     wfa_2_statistics[wfa_2_index] += 1;
//   }
//   tc.print("wfa_2:");
//
//   for (int i = 0; i < 40000; ++i) {
//     auto wfs_index = wfs();
//     wfs_statistics[wfs_index] += 1;
//   }
//   tc.print("wfs:");
//
//   for (int i = 0; i < 40000; ++i) {
//     auto wfb_index = wfb();
//     wfb_statistics[wfb_index] += 1;
//   }
//   tc.print("wfb:");
//
//   // break-point
//   int i = 0;
//   i += 1;
// }