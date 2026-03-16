#include "nikitina_v_hoar_sort_batcher/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace nikitina_v_hoar_sort_batcher {

namespace {
void CompareSplit(std::vector<int> &arr, int start1, int len1, int start2, int len2) {
  std::vector<int> temp(len1 + len2);
  int i = start1, j = start2, k = 0;

  while (i < start1 + len1 && j < start2 + len2) {
    if (arr[i] <= arr[j]) {
      temp[k++] = arr[i++];
    } else {
      temp[k++] = arr[j++];
    }
  }
  while (i < start1 + len1) {
    temp[k++] = arr[i++];
  }
  while (j < start2 + len2) {
    temp[k++] = arr[j++];
  }

  for (int idx = 0; idx < len1; ++idx) {
    arr[start1 + idx] = temp[idx];
  }
  for (int idx = 0; idx < len2; ++idx) {
    arr[start2 + idx] = temp[len1 + idx];
  }
}
}  // namespace

HoareSortBatcherOMP::HoareSortBatcherOMP(const InType &in) : input_(in) {}

bool HoareSortBatcherOMP::ValidationImpl() {
  return true;
}

bool HoareSortBatcherOMP::PreProcessingImpl() {
  output_ = input_;
  return true;
}

bool HoareSortBatcherOMP::RunImpl() {
  int n = output_.size();
  if (n <= 1) {
    return true;
  }

  int max_threads = omp_get_max_threads();
  int t = 1;
  while (t * 2 <= max_threads && t * 2 <= n) {
    t *= 2;
  }

  if (t == 1) {
    std::sort(output_.begin(), output_.end());
    return true;
  }

  std::vector<int> offsets(t + 1, 0);
  int base_chunk = n / t;
  int rem = n % t;
  for (int i = 0; i < t; ++i) {
    offsets[i + 1] = offsets[i] + base_chunk + (i < rem ? 1 : 0);
  }

#pragma omp parallel num_threads(t)
  {
    int tid = omp_get_thread_num();
    std::sort(output_.begin() + offsets[tid], output_.begin() + offsets[tid + 1]);
  }

  for (int p = 1; p < t; p *= 2) {
    for (int k = p; k > 0; k /= 2) {
      std::vector<std::pair<int, int>> pairs;
      for (int j = k % p; j + k < t; j += (k * 2)) {
        for (int i = 0; i < std::min(k, t - j - k); i++) {
          if ((j + i) / (p * 2) == (j + i + k) / (p * 2)) {
            pairs.push_back({j + i, j + i + k});
          }
        }
      }

      int num_pairs = static_cast<int>(pairs.size());

#pragma omp parallel for num_threads(t)
      for (int idx = 0; idx < num_pairs; ++idx) {
        int a = pairs[idx].first;
        int b = pairs[idx].second;
        CompareSplit(output_, offsets[a], offsets[a + 1] - offsets[a], offsets[b], offsets[b + 1] - offsets[b]);
      }
    }
  }

  return true;
}

bool HoareSortBatcherOMP::PostProcessingImpl() {
  return true;
}

}  // namespace nikitina_v_hoar_sort_batcher
