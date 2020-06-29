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
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_MLMD_BENCH_WORKLOAD_H
#define ML_METADATA_TOOLS_MLMD_BENCH_MLMD_BENCH_WORKLOAD_H

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/variant.h"

#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/tools/mlmd_bench/mlmd_bench_stats.h"
#include "ml_metadata/tools/mlmd_bench/mlmd_bench_watch.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// A base class for the Workload class.
class WorkloadBase {
 public:
  WorkloadBase() = default;
  virtual ~WorkloadBase() = default;

  // Prepares a list of work items in memory. It may read db to prepare work
  // items.
  virtual tensorflow::Status SetUp() = 0;

  // Measure performance for the workload operation on individual work item on
  // MLMD. Only this function will be counted towards performance measurement.
  virtual tensorflow::Status RunOp(int i, Stats::OpStats& op_stats,
                                   Watch& watch) = 0;

  // Cleans the list of work items.
  virtual tensorflow::Status TearDown() = 0;

  // Get the specification for current workload.
  virtual std::string GetSpecification() = 0;

  // Get the number of operations for current workload.
  virtual int GetNumOps() = 0;

  // Get the set up status for current workload.
  virtual bool GetSetUpStatus() = 0;
};

// A base class for all specific workloads (FillTypes, FillNodes, ...).
// It is a template class where WorkItemType defines the type for the list of
// work items prepared in SetUp().
template <typename WorkItemType>
class Workload : public WorkloadBase {
 protected:
  std::string specification_;
  int num_ops_;
  bool is_setup_;
  // The list of work items(PutArtifactTypeRequest, PutExecutionRequest...)
  // prepared by the SetUp().
  std::vector<WorkItemType> setup_work_items_;
  // The list of transferred bytes for each work item.
  std::vector<int> setup_work_items_bytes_;

 public:
  Workload();
  ~Workload() override = default;

  tensorflow::Status SetUp();

  // The function called inside the SetUp(), it will be implemented inside each
  // specific workload(FillTypes, FillNodes, ...) according to their semantics.
  virtual tensorflow::Status SetUpImpl();

  tensorflow::Status RunOp(int i, Stats::OpStats& op_stats, Watch& watch);

  // The function called inside the RunOp(), it will be implemented inside each
  // specific workload(FillTypes, FillNodes, ...) according to their semantics.
  virtual tensorflow::Status RunOpImpl(int i);

  tensorflow::Status TearDown();

  // The function called inside the TearDown(), it will be implemented inside
  // each specific workload(FillTypes, FillNodes, ...) according to their
  // semantics.
  virtual tensorflow::Status TearDownImpl();

  std::string GetSpecification();

  int GetNumOps();

  bool GetSetUpStatus();
};

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_MLMD_BENCH_WORKLOAD_H
