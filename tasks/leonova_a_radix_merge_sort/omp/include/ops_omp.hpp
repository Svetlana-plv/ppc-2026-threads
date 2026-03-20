#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "leonova_a_radix_merge_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace leonova_a_radix_merge_sort {

class LeonovaARadixMergeSortOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }

  explicit LeonovaARadixMergeSortOMP(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void RadixMergeSort(std::vector<int64_t> &arr, size_t left, size_t right);
  static void RadixSort(std::vector<int64_t> &arr, size_t left, size_t right);
  static void SimpleMerge(std::vector<int64_t> &arr, size_t left, size_t mid, size_t right);

  static uint64_t ToUnsigned(int64_t value);

  static constexpr size_t kRadixThreshold = 131072;
  static constexpr int kByteSize = 8;
  static constexpr int kNumBytes = 8;
  static constexpr int kNumCounters = 256;
  static constexpr uint64_t kSignBitMask = 0x8000000000000000ULL;

  using CounterRow = std::vector<size_t>;
  using CounterTable = std::vector<CounterRow>;

  static inline uint64_t ToUnsignedValue(int64_t value);
  static inline void FillUnsignedKeys(const std::vector<int64_t> &arr, size_t left, size_t size,
                                      std::vector<uint64_t> &keys);
  static inline void CopyTempToArray(std::vector<int64_t> &arr, size_t left, std::vector<int64_t> &temp_arr,
                                     std::vector<uint64_t> &keys, std::vector<uint64_t> &temp_keys);

  static inline void ResetLocalCounts(LeonovaARadixMergeSortOMP::CounterTable &local_counts);
  static inline void BuildThreadOffsets(const LeonovaARadixMergeSortOMP::CounterTable &local_counts,
                                        size_t thread_count, LeonovaARadixMergeSortOMP::CounterTable &local_offsets);
  static inline void CountByteValues(const std::vector<uint64_t> &keys, size_t size, int shift,
                                     LeonovaARadixMergeSortOMP::CounterTable &local_counts);
  static inline void ScatterByte(const std::vector<uint64_t> &keys, const std::vector<int64_t> &arr, size_t left,
                                 size_t size, int shift, LeonovaARadixMergeSortOMP::CounterTable &local_offsets,
                                 std::vector<int64_t> &temp_arr, std::vector<uint64_t> &temp_keys);
};

}  // namespace leonova_a_radix_merge_sort
