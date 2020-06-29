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
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_MLMD_BENCH_FILL_TYPES_H
#define ML_METADATA_TOOLS_MLMD_BENCH_MLMD_BENCH_FILL_TYPES_H

#include <random>

#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/variant.h"
#include "google/protobuf/text_format.h"

#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/tools/mlmd_bench/mlmd_bench_stats.h"
#include "ml_metadata/tools/mlmd_bench/mlmd_bench_workload.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
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
 private:
  // Random generator for generating random type names.
  std::minstd_rand0 gen_;
  // A list of generated unique type names.
  std::vector<std::string> types_name_;
  // MetadataStore for current workload.
  std::unique_ptr<MetadataStore>* store_ptr_;
  // Workload configurations specified by the users.
  WorkloadConfig workload_config_;

 public:
  FillTypes(std::unique_ptr<MetadataStore>* store_ptr,
            const WorkloadConfig& workload_config);
  ~FillTypes() override = default;

  // Set up implementation for FillTypes workload, it will not be included into
  // the performance evaluation.
  tensorflow::Status SetUpImpl();

  // The real operation being performed against the database.
  // This function will be counted towards performance measurement.
  tensorflow::Status RunOpImpl(int i);

  // Tear down implementation for FillTypes, it will not be included into
  // the performance evaluation.
  tensorflow::Status TearDownImpl(Stats& thread_stats);

  // Get the list of work items.
  std::vector<FillTypeWorkItemType> GetWorkItem();

  // Get the list of work item bytes.
  std::vector<int> GetWorkItemBytes();

  // Get the list of generated unique type names.
  std::vector<std::string> GetTypesName();
};

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_MLMD_BENCH_FILL_TYPES_H
