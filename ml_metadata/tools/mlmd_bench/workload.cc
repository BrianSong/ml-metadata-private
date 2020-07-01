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
#include "ml_metadata/tools/mlmd_bench/workload.h"

namespace ml_metadata {

// Set the num_ops and is_setup to their initial value.
template <typename WorkItemType>
Workload<WorkItemType>::Workload() {
  is_setup_ = false;
}

template <typename WorkItemType>
tensorflow::Status Workload<WorkItemType>::SetUp(
    MetadataStore* set_up_store_ptr) {
  TF_RETURN_IF_ERROR(SetUpImpl(set_up_store_ptr));
  // Set the is_setup to true to ensure execution sequence.
  is_setup_ = true;
  return tensorflow::Status::OK();
}

template <typename WorkItemType>
tensorflow::Status Workload<WorkItemType>::SetUpImpl(
    MetadataStore* set_up_store_ptr) {
  return tensorflow::errors::Unimplemented(
      "SetUpImpl() for Workload class is not implemented!");
}

template <typename WorkItemType>
tensorflow::Status Workload<WorkItemType>::RunOp(int i, Watch watch,
                                                 MetadataStore* store_ptr,
                                                 OpStats& op_stats) {
  // Check is_setup to ensure execution sequence.
  if (!is_setup_) {
    return tensorflow::errors::FailedPrecondition("Set up is not finished!");
  }
  // Use a watch to calculate the elapsed time of each RunOpImpl().
  watch.Start();
  TF_RETURN_IF_ERROR(RunOpImpl(i, store_ptr));
  watch.End();
  // Each operation will have an op_stats to record its statistic using the
  // execution.
  op_stats.elapsed_micros = watch.GetElaspedTimeInMicroS();
  op_stats.transferred_types = work_items_bytes_[i];
  return tensorflow::Status::OK();
}

template <typename WorkItemType>
tensorflow::Status Workload<WorkItemType>::RunOpImpl(int i,
                                                     MetadataStore* store_ptr) {
  return tensorflow::errors::Unimplemented(
      "RunOpImpl() for Workload class is not implemented!");
}

template <typename WorkItemType>
tensorflow::Status Workload<WorkItemType>::TearDown() {
  // Check is_setup to ensure execution sequence.
  if (!is_setup_) {
    return tensorflow::errors::FailedPrecondition("Set up is not finished!");
  }
  TF_RETURN_IF_ERROR(TearDownImpl());
  return tensorflow::Status::OK();
}

template <typename WorkItemType>
tensorflow::Status Workload<WorkItemType>::TearDownImpl() {
  return tensorflow::errors::Unimplemented(
      "TearDownImpl() for Workload class is not implemented!");
}

template <typename WorkItemType>
int Workload<WorkItemType>::num_ops() {
  return work_items_.size();
}

// Avoid link error.
template class Workload<absl::variant<
    PutArtifactTypeRequest, PutExecutionTypeRequest, PutContextTypeRequest>>;

// Avoid link error.
template class Workload<std::string>;

}  // namespace ml_metadata
