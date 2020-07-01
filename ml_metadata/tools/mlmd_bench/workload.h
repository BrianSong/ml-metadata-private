/* Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_WORKLOAD_H
#define ML_METADATA_TOOLS_MLMD_BENCH_WORKLOAD_H

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/variant.h"

#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/stats.h"
#include "ml_metadata/tools/mlmd_bench/watch.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// A base class for all Workloads. Each workload subclass takes a config.,
// creates the datasets, then for each work item of the dataset, it runs an
// operation against MLMD, and measures its performance.
class WorkloadBase {
 public:
  WorkloadBase() = default;
  virtual ~WorkloadBase() = default;

  // Prepares a list of work items in memory. It may read db to prepare work
  // items.
  virtual tensorflow::Status SetUp(MetadataStore* set_up_store_ptr) = 0;

  // Measure performance for the workload operation on individual work item on
  // MLMD.
  virtual tensorflow::Status RunOp(int i, Watch watch, MetadataStore* store_ptr,
                                   OpStats& op_stats) = 0;

  // Cleans the list of work items and related resources.
  virtual tensorflow::Status TearDown() = 0;

  // Get the number of operations for current workload.
  virtual int num_ops() = 0;
};

// A base class for all specific workloads (FillTypes, FillNodes, ...).
// It is a template class where WorkItemType defines the type for the list of
// work items prepared in SetUp().
template <typename WorkItemType>
class Workload : public WorkloadBase {
 public:
  Workload();
  virtual ~Workload() = default;

  // Prepares data related to a workload. It may read the information in the
  // store.
  // The method must run before RunOp() to isolate the data preparation
  // operations with the operation to be measured. The subclass should implement
  // SetUpImpl(). The given store should be not null and connected. Returns
  // detailed error if query executions failed.
  tensorflow::Status SetUp(MetadataStore* set_up_store_ptr) final;

  // Runs the operation of the workload on a work item i on the store.
  // The operation is measured and kept in op_stats. The subclass should
  // implement RunOpImpl(), and does not perform irrelevant operations to avoid
  // being counted in op_stats. Returns Failed Precondition error, if SetUp() is
  // not finished before running the operation. Returns detailed error if query
  // execution failed.
  tensorflow::Status RunOp(int i, Watch watch, MetadataStore* store_ptr,
                           OpStats& op_stats) final;

  // Cleans the list of work items and related resources.
  // The cleaning operation will not be included for performance measurement.
  // The subclass should implement RunOpImpl(). Returns Failed Precondition
  // error, if SetUp() is not finished before running the operation. Returns
  // detailed error if query execution failed.
  tensorflow::Status TearDown() final;

  int num_ops() final;

 protected:
  // The function called inside the SetUp(), it will be implemented inside each
  // specific workload(FillTypes, FillNodes, ...) according to their semantics.
  virtual tensorflow::Status SetUpImpl(MetadataStore* set_up_store_ptr);

  // The function called inside the RunOp(), it will be implemented inside each
  // specific workload(FillTypes, FillNodes, ...) according to their semantics.
  virtual tensorflow::Status RunOpImpl(int i, MetadataStore* store_ptr);

  // The function called inside the TearDown(), it will be implemented inside
  // each specific workload(FillTypes, FillNodes, ...) according to their
  // semantics.
  virtual tensorflow::Status TearDownImpl();

  bool is_setup_;
  // The work items for a workload. It is created in SetUp(), and each RunOp
  // processes one work item.
  std::vector<WorkItemType> work_items_;
  // The list of transferred bytes for each work item.
  std::vector<int> work_items_bytes_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_WORKLOAD_H
