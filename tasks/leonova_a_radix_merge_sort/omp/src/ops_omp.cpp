#include "leonova_a_radix_merge_sort/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "leonova_a_radix_merge_sort/common/include/common.hpp"
#include "util/include/util.hpp"

namespace leonova_a_radix_merge_sort {

inline uint64_t LeonovaARadixMergeSortOMP::ToUnsignedValue(int64_t value) {
  return static_cast<uint64_t>(value) ^ kSignBitMask;
}

inline void LeonovaARadixMergeSortOMP::ResetLocalCounts(CounterTable &local_counts) {
  for (auto &counter : local_counts) {
    std::ranges::fill(counter, 0);
  }
}

inline void LeonovaARadixMergeSortOMP::BuildThreadOffsets(const CounterTable &local_counts, size_t thread_count,
                                                          CounterTable &local_offsets) {
  std::vector<size_t> bucket_totals(kNumCounters, 0);

  for (size_t bucket = 0; bucket < kNumCounters; ++bucket) {
    size_t total = 0;
    for (size_t thread = 0; thread < thread_count; ++thread) {
      total += local_counts[thread][bucket];
    }
    bucket_totals[bucket] = total;
  }

  size_t prefix = 0;
  for (size_t bucket = 0; bucket < kNumCounters; ++bucket) {
    const size_t count = bucket_totals[bucket];
    bucket_totals[bucket] = prefix;
    prefix += count;
  }

  for (size_t bucket = 0; bucket < kNumCounters; ++bucket) {
    size_t current = bucket_totals[bucket];
    for (size_t thread = 0; thread < thread_count; ++thread) {
      local_offsets[thread][bucket] = current;
      current += local_counts[thread][bucket];
    }
  }
}

inline void LeonovaARadixMergeSortOMP::FillUnsignedKeys(const std::vector<int64_t> &arr, size_t left, size_t size,
                                                        std::vector<uint64_t> &keys) {
#pragma omp for schedule(static)
  for (size_t index = 0; index < size; ++index) {
    keys[index] = ToUnsignedValue(arr[left + index]);
  }
}

inline void LeonovaARadixMergeSortOMP::CountByteValues(const std::vector<uint64_t> &keys, size_t size, int shift,
                                                       CounterTable &local_counts) {
#pragma omp for schedule(static)
  for (size_t index = 0; index < size; ++index) {
    const auto byte_val = static_cast<size_t>((keys[index] >> shift) & 0xFFU);
    const auto thread_id = static_cast<size_t>(omp_get_thread_num());
    ++local_counts[thread_id][byte_val];
  }
}

inline void LeonovaARadixMergeSortOMP::ScatterByte(const std::vector<uint64_t> &keys, const std::vector<int64_t> &arr,
                                                   size_t left, size_t size, int shift, CounterTable &local_offsets,
                                                   std::vector<int64_t> &temp_arr, std::vector<uint64_t> &temp_keys) {
  const auto thread_id = static_cast<size_t>(omp_get_thread_num());
  auto &thread_offsets = local_offsets[thread_id];

#pragma omp for schedule(static)
  for (size_t index = 0; index < size; ++index) {
    const auto byte_val = static_cast<size_t>((keys[index] >> shift) & 0xFFU);
    const size_t pos = thread_offsets[byte_val]++;
    temp_arr[pos] = arr[left + index];
    temp_keys[pos] = keys[index];
  }
}

inline void LeonovaARadixMergeSortOMP::CopyTempToArray(std::vector<int64_t> &arr, size_t left,
                                                       std::vector<int64_t> &temp_arr, std::vector<uint64_t> &keys,
                                                       std::vector<uint64_t> &temp_keys) {
#pragma omp single
  {
    std::ranges::copy(temp_arr, arr.data() + left);
    keys.swap(temp_keys);
  }
}

LeonovaARadixMergeSortOMP::LeonovaARadixMergeSortOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int64_t>(GetInput().size());
}

bool LeonovaARadixMergeSortOMP::ValidationImpl() {
  return !GetInput().empty();
}

bool LeonovaARadixMergeSortOMP::PreProcessingImpl() {
  return true;
}

bool LeonovaARadixMergeSortOMP::RunImpl() {
  if (!ValidationImpl()) {
    return false;
  }

  GetOutput() = GetInput();
  if (GetOutput().size() > 1) {
    RadixMergeSort(GetOutput(), 0, GetOutput().size());
  }

  return true;
}

bool LeonovaARadixMergeSortOMP::PostProcessingImpl() {
  return true;
}

uint64_t LeonovaARadixMergeSortOMP::ToUnsigned(int64_t value) {
  return ToUnsignedValue(value);
}

void LeonovaARadixMergeSortOMP::RadixSort(std::vector<int64_t> &arr, size_t left, size_t right) {
  const size_t size = right - left;
  if (size <= 1) {
    return;
  }

  const int requested_threads = std::max(1, ppc::util::GetNumThreads());
  const int omp_threads = std::max(1, std::min(requested_threads, static_cast<int>(size)));
  const auto thread_count = static_cast<size_t>(omp_threads);

  std::vector<uint64_t> keys(size);
  std::vector<int64_t> temp_arr(size);
  std::vector<uint64_t> temp_keys(size);
  CounterTable local_counts(thread_count, CounterRow(kNumCounters, 0));
  CounterTable local_offsets(thread_count, CounterRow(kNumCounters, 0));

#pragma omp parallel default(none) shared(arr, keys, temp_arr, temp_keys, local_counts, local_offsets, left, size, \
                                              omp_threads, thread_count) num_threads(omp_threads)
  {
    FillUnsignedKeys(arr, left, size, keys);

    for (int byte_pos = 0; byte_pos < kNumBytes; ++byte_pos) {
      const int shift = byte_pos * kByteSize;

#pragma omp single
      {
        ResetLocalCounts(local_counts);
      }

#pragma omp barrier

      CountByteValues(keys, size, shift, local_counts);

#pragma omp single
      {
        BuildThreadOffsets(local_counts, thread_count, local_offsets);
      }

#pragma omp barrier

      ScatterByte(keys, arr, left, size, shift, local_offsets, temp_arr, temp_keys);

      CopyTempToArray(arr, left, temp_arr, keys, temp_keys);

#pragma omp barrier
    }
  }
}

void LeonovaARadixMergeSortOMP::SimpleMerge(std::vector<int64_t> &arr, size_t left, size_t mid, size_t right) {
  const size_t left_size = mid - left;
  const size_t right_size = right - mid;
  std::vector<int64_t> merged(left_size + right_size);

  size_t i = 0;
  size_t j = 0;
  size_t k = 0;

  while (i < left_size && j < right_size) {
    if (arr[left + i] <= arr[mid + j]) {
      merged[k++] = arr[left + i++];
    } else {
      merged[k++] = arr[mid + j++];
    }
  }

  while (i < left_size) {
    merged[k++] = arr[left + i++];
  }

  while (j < right_size) {
    merged[k++] = arr[mid + j++];
  }

  std::ranges::copy(merged, arr.data() + left);
}

void LeonovaARadixMergeSortOMP::RadixMergeSort(std::vector<int64_t> &arr, size_t left, size_t right) {
  struct SortTask {
    size_t left;
    size_t right;
    bool sorted;
  };

  std::vector<SortTask> stack;
  stack.reserve(128);
  stack.push_back({left, right, false});

  while (!stack.empty()) {
    SortTask current = stack.back();
    stack.pop_back();

    const size_t size = current.right - current.left;
    if (size <= 1) {
      continue;
    }

    if (size <= kRadixThreshold) {
      RadixSort(arr, current.left, current.right);
      continue;
    }

    if (!current.sorted) {
      const size_t mid = current.left + (size / 2);
      stack.push_back({current.left, current.right, true});
      stack.push_back({mid, current.right, false});
      stack.push_back({current.left, mid, false});
    } else {
      const size_t mid = current.left + (size / 2);
      SimpleMerge(arr, current.left, mid, current.right);
    }
  }
}

}  // namespace leonova_a_radix_merge_sort
