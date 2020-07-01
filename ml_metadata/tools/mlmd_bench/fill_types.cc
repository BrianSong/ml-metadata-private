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
#include "ml_metadata/tools/mlmd_bench/fill_types.h"

namespace ml_metadata {

namespace {

template <typename Type>
void GenerateRandomType(Type* type, std::uniform_int_distribution<int>& uid,
                        std::minstd_rand0& gen, int& curr_bytes) {
  type->set_name(absl::StrCat(rand()));
  curr_bytes += type->name().size();
  const int num_properties = uid(gen);
  for (int i = 0; i < num_properties; i++) {
    (*type->mutable_properties())[absl::StrCat("p-", i)] = STRING;
    curr_bytes += absl::StrCat("p-", i).size();
  }
}

// template <typename Type>
// tensorflow::Status CheckTypeExistOrNot(MetadataStore* set_up_store_ptr) {}

}  // namespace

// FillTypes constructor for setting up its configurations.
FillTypes::FillTypes(const FillTypesConfig& fill_types_config, int num_ops)
    : fill_types_config_(fill_types_config), num_ops_(num_ops) {}

tensorflow::Status FillTypes::SetUpImpl(MetadataStore* set_up_store_ptr) {
  std::fprintf(stderr, "Setting up ...");
  std::fflush(stderr);

  int op = 0;
  int curr_bytes = 0;
  // Uniform distribution describing the number of properties for each
  // generated types.
  UniformDistribution num_properties = fill_types_config_.num_properties();
  int min = num_properties.a();
  int max = num_properties.b();
  std::uniform_int_distribution<int> uid{min, max};
  // The seed for the random generator will be the time when the FillTypes is
  // created.
  std::minstd_rand0 gen;
  gen.seed(absl::ToUnixMillis(absl::Now()));

  while (op < num_ops_) {
    curr_bytes = 0;
    switch (fill_types_config_.specification()) {
      case (FillTypesConfig::ArtifactType): {
        PutArtifactTypeRequest put_request;
        GenerateRandomType<ArtifactType>(put_request.mutable_artifact_type(),
                                         uid, gen, curr_bytes);
        work_items_.push_back(put_request);
        work_items_bytes_.push_back(curr_bytes);
        op++;
        break;
      }
      case (FillTypesConfig::ExecutionType): {
        PutExecutionTypeRequest put_request;
        GenerateRandomType<ExecutionType>(put_request.mutable_execution_type(),
                                          uid, gen, curr_bytes);
        work_items_.push_back(put_request);
        work_items_bytes_.push_back(curr_bytes);
        op++;
        break;
      }
      case (FillTypesConfig::ContextType): {
        PutContextTypeRequest put_request;
        GenerateRandomType<ContextType>(put_request.mutable_context_type(), uid,
                                        gen, curr_bytes);
        work_items_.push_back(put_request);
        work_items_bytes_.push_back(curr_bytes);
        op++;
        break;
      }
      default:
        return tensorflow::errors::InvalidArgument("Wrong specification!");
    }
  }
  return tensorflow::Status::OK();
}

// Executing the work item prepared in SetUpImpl().
tensorflow::Status FillTypes::RunOpImpl(int i, MetadataStore* store_ptr) {
  switch (fill_types_config_.specification()) {
    case (FillTypesConfig::ArtifactType): {
      PutArtifactTypeRequest put_request =
          absl::get<PutArtifactTypeRequest>(work_items_[i]);
      PutArtifactTypeResponse put_response;
      return (store_ptr)->PutArtifactType(put_request, &put_response);
    }
    case (FillTypesConfig::ExecutionType): {
      PutExecutionTypeRequest put_request =
          absl::get<PutExecutionTypeRequest>(work_items_[i]);
      PutExecutionTypeResponse put_response;
      return (store_ptr)->PutExecutionType(put_request, &put_response);
    }
    case (FillTypesConfig::ContextType): {
      PutContextTypeRequest put_request =
          absl::get<PutContextTypeRequest>(work_items_[i]);
      PutContextTypeResponse put_response;
      return (store_ptr)->PutContextType(put_request, &put_response);
    }
    default:
      return tensorflow::errors::InvalidArgument("Wrong specification!");
  }
  return tensorflow::errors::InvalidArgument(
      "Cannot execute the query due to wrong specification!");
}

tensorflow::Status FillTypes::TearDownImpl() {
  work_items_.clear();
  work_items_bytes_.clear();
  types_name_.clear();
  return tensorflow::Status::OK();
}

std::vector<FillTypeWorkItemType> FillTypes::setup_work_items() {
  return work_items_;
}

std::vector<int> FillTypes::work_items_bytes() { return work_items_bytes_; }

std::vector<std::string> FillTypes::types_name() { return types_name_; }

FillTypesConfig FillTypes::config() { return fill_types_config_; }

}  // namespace ml_metadata
