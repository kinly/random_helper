#include "random.hpp"
#include <chrono>
#include <iostream>

using namespace inlay::random;

template <typename Sampler>
void benchmark(const std::string& name, Sampler& sampler, size_t N) {
  auto start = std::chrono::steady_clock::now();
  volatile int sink = 0;
  for (size_t i = 0; i < N; ++i) {
    sink += sampler();
  }
  auto end = chrono::steady_clock::now();
  std::chrono::duration<double, milli> dur = end - start;
  std::cout << name << ": " << dur.count() << " ms\n";
}

int main() {
  const size_t N = 1'000'000;
  std::vector<int> values(10);
  std::iota(values.begin(), values.end(), 1);
  std::vector<uint32_t> weights(values.begin(), values.end());

  weight_faster_alias<int> alias_sampler(values.begin(), values.end(), weights.begin(), weights.end());
  weight_faster_expansion<int> expansion_sampler(values.begin(), values.end(), weights.begin(), weights.end());
  weight_faster_binary<int> binary_sampler(values.begin(), values.end(), weights.begin(), weights.end());

  benchmark("Alias", alias_sampler, N);
  benchmark("Expansion", expansion_sampler, N);
  benchmark("Binary", binary_sampler, N);

  return 0;
}