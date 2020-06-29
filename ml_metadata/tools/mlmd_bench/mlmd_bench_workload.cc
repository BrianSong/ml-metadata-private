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
#include "ml_metadata/tools/mlmd_bench/mlmd_bench_workload.h"

namespace ml_metadata {

// Set the num_ops and is_setup to their initial value.
template <typename WorkItemType>
Workload<WorkItemType>::Workload() {
  num_ops_ = 0;
  is_setup_ = false;
}

template <typename WorkItemType>
tensorflow::Status Workload<WorkItemType>::SetUp() {
  // Set the is_setup to true to ensure execution sequence.
  is_setup_ = true;
  return SetUpImpl();
}

template <typename WorkItemType>
tensorflow::Status Workload<WorkItemType>::SetUpImpl() {
  return tensorflow::Status::OK();
}

template <typename WorkItemType>
tensorflow::Status Workload<WorkItemType>::RunOp(int i,
                                                 Stats::OpStats& op_stats,
                                                 Watch& watch) {
  // Check is_setup to ensure execution sequence.
  if (!is_setup_) {
    return tensorflow::errors::FailedPrecondition("Set up is not finished!");
  }
  // Use a watch to calculate the elapsed time of each RunOpImpl().
  watch.Start();
  if (RunOpImpl(i).ok()) {
    watch.End();
    // Each operation will have an op_stats to record its statistic using the
    // execution.
    op_stats.elapsed_micros = watch.GetElaspedTimeInMicroS();
    op_stats.transferred_types = setup_work_items_bytes_[i];
  } else {
    return tensorflow::errors::Unimplemented(
        "Can not perform current operation!");
  }
  return tensorflow::Status::OK();
}

template <typename WorkItemType>
tensorflow::Status Workload<WorkItemType>::RunOpImpl(int i) {
  return tensorflow::Status::OK();
}

template <typename WorkItemType>
tensorflow::Status Workload<WorkItemType>::TearDown() {
  return TearDownImpl();
}

template <typename WorkItemType>
tensorflow::Status Workload<WorkItemType>::TearDownImpl() {
  return tensorflow::Status::OK();
}

template <typename WorkItemType>
std::string Workload<WorkItemType>::GetSpecification() {
  return specification_;
}

template <typename WorkItemType>
int Workload<WorkItemType>::GetNumOps() {
  return num_ops_;
}

template <typename WorkItemType>
bool Workload<WorkItemType>::GetSetUpStatus() {
  return is_setup_;
}

// Avoid link error.
template class Workload<absl::variant<
    PutArtifactTypeRequest, PutExecutionTypeRequest, PutContextTypeRequest>>;

}  // namespace ml_metadata
