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
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_FILL_TYPES_WORKLOAD_H
#define ML_METADATA_TOOLS_MLMD_BENCH_FILL_TYPES_WORKLOAD_H

#include "absl/types/variant.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/types.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/workload.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// Defines a FillTypeWorkItemType that can be PutArtifactTypeRequest /
// PutExecutionTypeRequest / PutContextTypeRequest.
using FillTypeWorkItemType =
    absl::variant<PutArtifactTypeRequest, PutExecutionTypeRequest,
                  PutContextTypeRequest>;
// A specific workload for creating and updating types: ArtifactTypes /
// ExecutionTypes / ContextTypes.
class FillTypes : public Workload<FillTypeWorkItemType> {
 public:
  FillTypes(const FillTypesConfig& fill_types_config, int64 num_operations);
  ~FillTypes() override = default;

 protected:
  // Specific implementation of SetUpImpl() for FillTypes workload according to
  // its semantic. The detail implementation will depend on whether the current
  // FillTypes is for inserting or updating types.
  // For inserting, it will generate the list of work items(FillTypesRequests)
  // by generating a type name and the number of properties for each type
  // w.r.t. the uniform distribution.
  // Returns detailed error if query executions failed.
  tensorflow::Status SetUpImpl(MetadataStore* store) final;

  // Specific implementation of RunOpImpl() for FillTypes workload according to
  // its semantic. Runs the work items(FillTypesRequests) on the store. Returns
  // detailed error if query executions failed.
  tensorflow::Status RunOpImpl(int64 i, MetadataStore* store) final;

  // Specific implementation of TearDownImpl() for FillTypes workload according
  // to its semantic. Cleans the work_items_.
  tensorflow::Status TearDownImpl() final;

  // Gets the current workload's name, which is used in stats report for this
  // workload.
  std::string GetName() final;

 private:
  // Workload configurations specified by the users.
  const FillTypesConfig fill_types_config_;
  // Number of operations for the current workload.
  const int64 num_operations_;
  // String for indicating the name of current workload instance.
  std::string name_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_FILL_TYPES_WORKLOAD_H
