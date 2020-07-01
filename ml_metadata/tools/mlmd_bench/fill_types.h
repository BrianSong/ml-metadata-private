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
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_FILL_TYPES_H
#define ML_METADATA_TOOLS_MLMD_BENCH_FILL_TYPES_H

#include <random>

#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/variant.h"
#include "google/protobuf/text_format.h"

#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "ml_metadata/tools/mlmd_bench/stats.h"
#include "ml_metadata/tools/mlmd_bench/workload.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

// Define a FillTypeWorkItemType that can be PutArtifactTypeRequest /
// PutExecutionTypeRequest / PutContextTypeRequest.
using FillTypeWorkItemType =
    absl::variant<PutArtifactTypeRequest, PutExecutionTypeRequest,
                  PutContextTypeRequest>;

// A specific workload for creating and updating types: ArtifactTypes /
// ExecutionTypes / ContextTypes
class FillTypes : public Workload<FillTypeWorkItemType> {
 public:
  FillTypes(const FillTypesConfig& fill_types_config, int num_ops);
  ~FillTypes() override = default;

  // Get the list of work items.
  std::vector<FillTypeWorkItemType> setup_work_items();

  // Get the list of work item bytes.
  std::vector<int> work_items_bytes();

  // Get the list of generated unique type names.
  std::vector<std::string> types_name();

  FillTypesConfig config();

 protected:
  // Set up implementation for FillTypes workload, it will not be included into
  // the performance evaluation.
  tensorflow::Status SetUpImpl(MetadataStore* set_up_store_ptr) final;

  // The real operation being performed against the database.
  // This function will be counted towards performance measurement.
  tensorflow::Status RunOpImpl(int i, MetadataStore* store_ptr) final;

  // Tear down implementation for FillTypes, it will not be included into
  // the performance evaluation.
  tensorflow::Status TearDownImpl() final;

 private:
  // A list of generated unique type names.
  std::vector<std::string> types_name_;
  // Workload configurations specified by the users.
  FillTypesConfig fill_types_config_;
  // Number of operations executing the current workload.
  int num_ops_;
};

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_FILL_TYPES_H
