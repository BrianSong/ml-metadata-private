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
#include "ml_metadata/tools/mlmd_bench/fill_types_workload.h"

#include <random>
#include <set>
#include <vector>

#include "absl/strings/substitute.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {
namespace {

// A set used to ensure each type name generated is unique.
std::set<std::string> unique_name_checker;

// A template function where the Type can be ArtifactType / ExecutionType /
// ContextType.
// Generates random type names and the number of properties for
// each type w.r.t. the uniform distribution. Uses the unique_name_checker to
// ensure each random type name generated is unique. If a new generated type
// name exists inside the unique_name_checker, return false to perform rejection
// sampling. Otherwise, return true.
template <typename Type>
bool GenerateRandomType(std::uniform_int_distribution<int64>& uniform_dist,
                        std::minstd_rand0& gen, Type* type, int64& curr_bytes) {
  // The random type name will be a random number.
  type->set_name(absl::StrCat(rand()));
  // The curr_bytes records the total transferred bytes for executing each work
  // item.
  curr_bytes += type->name().size();
  // Generates the number of properties for each type
  // w.r.t. the uniform distribution
  const int64 num_properties = uniform_dist(gen);
  for (int64 i = 0; i < num_properties; i++) {
    (*type->mutable_properties())[absl::StrCat("p-", i)] = STRING;
    curr_bytes += absl::StrCat("p-", i).size();
  }
  // Uses unique_name_checker to check whether the current generated random type
  // name is unique.
  if (unique_name_checker.find(type->name()) == unique_name_checker.end()) {
    unique_name_checker.insert(type->name());
    return true;
  }
  return false;
}

// A template function where the Type can be ArtifactType / ExecutionType /
// ContextType.
// Takes an existed type and generates a update type accordingly. The updated
// type will have some new fields added and the number of new added fields will
// be generated w.r.t. the uniform distribution.
template <typename Type>
void UpdateType(std::uniform_int_distribution<int64>& uniform_dist,
                std::minstd_rand0& gen, const Type& existed_type,
                Type* updated_type, int64& curr_bytes) {
  // Except the new added fields, update_type will the same as existed_type.
  *updated_type = existed_type;
  curr_bytes += existed_type.name().size();
  for (auto& pair : existed_type.properties()) {
    // pair.first is the property of existed_type.
    curr_bytes += pair.first.size();
  }
  const int64 num_properties = uniform_dist(gen);
  for (int64 i = 0; i < num_properties; i++) {
    (*updated_type->mutable_properties())[absl::StrCat("add_p-", i)] = STRING;
    curr_bytes += absl::StrCat("add_p-", i).size();
  }
}

}  // namespace

FillTypes::FillTypes(const FillTypesConfig& fill_types_config, int64 num_ops)
    : fill_types_config_(fill_types_config), num_ops_(num_ops) {
  switch (fill_types_config_.specification()) {
    case FillTypesConfig::ArtifactType: {
      name_ = "fill_artifact_type";
      break;
    }
    case FillTypesConfig::ExecutionType: {
      name_ = "fill_execution_type";
      break;
    }
    case FillTypesConfig::ContextType: {
      name_ = "fill_context_type";
      break;
    }
    default:
      LOG(ERROR) << "Wrong specification for FillTypes!";
  }
}

// GetTypeResponseType can be GetArtifactTypesResponse /
// GetExecutionTypesResponse / GetContextTypesResponse.
using GetTypeResponseType =
    absl::variant<GetArtifactTypesResponse, GetExecutionTypesResponse,
                  GetContextTypesResponse>;
tensorflow::Status FillTypes::SetUpImpl(MetadataStore* store) {
  LOG(INFO) << "Setting up ...";

  int64 curr_bytes = 0;
  // Uniform distribution that describes the number of properties for each
  // generated types.
  UniformDistribution num_properties = fill_types_config_.num_properties();
  int64 min = num_properties.a();
  int64 max = num_properties.b();
  std::uniform_int_distribution<int64> uniform_dist{min, max};
  // The seed for the random generator is the time when the FillTypes is
  // created.
  std::minstd_rand0 gen(absl::ToUnixMillis(absl::Now()));

  GetTypeResponseType get_response;
  if (fill_types_config_.update()) {
    name_ += "(update)";
    switch (fill_types_config_.specification()) {
      case FillTypesConfig::ArtifactType: {
        get_response.emplace<GetArtifactTypesResponse>();
        GetArtifactTypesRequest get_request;
        // Gets all existed artifact types from db.
        TF_RETURN_IF_ERROR(store->GetArtifactTypes(
            get_request, &(absl::get<GetArtifactTypesResponse>(get_response))));
        // Checks for invalid num_ops_. The num_ops_ cannot be greater than the
        // number of current existed artifact types in db.
        if (num_ops_ >
            (long long)absl::get<GetArtifactTypesResponse>(get_response)
                .artifact_types()
                .size()) {
          return tensorflow::errors::InvalidArgument(
              "There are not enough artifact types to update!");
        }
        break;
      }
      case FillTypesConfig::ExecutionType: {
        get_response.emplace<GetExecutionTypesResponse>();
        GetExecutionTypesRequest get_request;
        // Gets all existed execution types from db.
        TF_RETURN_IF_ERROR(store->GetExecutionTypes(
            get_request,
            &(absl::get<GetExecutionTypesResponse>(get_response))));
        // Checks for invalid num_ops_. The num_ops_ cannot be greater than the
        // number of current existed execution types in db.
        if (num_ops_ >
            (long long)absl::get<GetExecutionTypesResponse>(get_response)
                .execution_types()
                .size()) {
          return tensorflow::errors::InvalidArgument(
              "There are not enough execution types to update!");
        }
        break;
      }
      case FillTypesConfig::ContextType: {
        get_response.emplace<GetContextTypesResponse>();
        GetContextTypesRequest get_request;
        // Gets all existed context types from db.
        TF_RETURN_IF_ERROR(store->GetContextTypes(
            get_request, &(absl::get<GetContextTypesResponse>(get_response))));
        // Checks for invalid num_ops_. The num_ops_ cannot be greater than the
        // number of current existed context types in db.
        if (num_ops_ >
            (long long)absl::get<GetContextTypesResponse>(get_response)
                .context_types()
                .size()) {
          return tensorflow::errors::InvalidArgument(
              "There are not enough execution types to update!");
        }
        break;
      }
      default:
        return tensorflow::errors::InvalidArgument("Wrong specification!");
    }
  }

  int64 op = 0;
  while (op < num_ops_) {
    curr_bytes = 0;
    FillTypeWorkItemType put_request;
    switch (fill_types_config_.specification()) {
      case FillTypesConfig::ArtifactType: {
        put_request.emplace<PutArtifactTypeRequest>();
        // Update cases.
        if (fill_types_config_.update()) {
          // Sets can_add_fields to true.
          absl::get<PutArtifactTypeRequest>(put_request)
              .set_can_add_fields(true);
          UpdateType<ArtifactType>(
              uniform_dist, gen,
              absl::get<GetArtifactTypesResponse>(get_response)
                  .artifact_types()[op],
              absl::get<PutArtifactTypeRequest>(put_request)
                  .mutable_artifact_type(),
              curr_bytes);
          break;
        }
        // Insert Cases.
        // If GenerateRandomType() returns false, the new generated random type
        // name is not unique. Jumps to next iteration and regenerates again.
        if (!GenerateRandomType<ArtifactType>(
                uniform_dist, gen,
                absl::get<PutArtifactTypeRequest>(put_request)
                    .mutable_artifact_type(),
                curr_bytes)) {
          continue;
        }
        break;
      }
      case FillTypesConfig::ExecutionType: {
        put_request.emplace<PutExecutionTypeRequest>();
        // Update cases.
        if (fill_types_config_.update()) {
          // Sets can_add_fields to true.
          absl::get<PutExecutionTypeRequest>(put_request)
              .set_can_add_fields(true);
          UpdateType<ExecutionType>(
              uniform_dist, gen,
              absl::get<GetExecutionTypesResponse>(get_response)
                  .execution_types()[op],
              absl::get<PutExecutionTypeRequest>(put_request)
                  .mutable_execution_type(),
              curr_bytes);
          break;
        }
        // Insert Cases.
        // If GenerateRandomType() returns false, the new generated random type
        // name is not unique. Jumps to next iteration and regenerates again.
        if (!GenerateRandomType<ExecutionType>(
                uniform_dist, gen,
                absl::get<PutExecutionTypeRequest>(put_request)
                    .mutable_execution_type(),
                curr_bytes)) {
          continue;
        }
        break;
      }
      case FillTypesConfig::ContextType: {
        put_request.emplace<PutContextTypeRequest>();
        // Update cases.
        if (fill_types_config_.update()) {
          // Sets can_add_fields to true.
          absl::get<PutContextTypeRequest>(put_request)
              .set_can_add_fields(true);
          UpdateType<ContextType>(
              uniform_dist, gen,
              absl::get<GetContextTypesResponse>(get_response)
                  .context_types()[op],
              absl::get<PutContextTypeRequest>(put_request)
                  .mutable_context_type(),
              curr_bytes);
          break;
        }
        // Insert Cases.
        // If GenerateRandomType() returns false, the new generated random type
        // name is not unique. Jumps to next iteration and regenerates again.
        if (!GenerateRandomType<ContextType>(
                uniform_dist, gen,
                absl::get<PutContextTypeRequest>(put_request)
                    .mutable_context_type(),
                curr_bytes)) {
          continue;
        }
        break;
      }
      default:
        return tensorflow::errors::InvalidArgument("Wrong specification!");
    }
    work_items_.emplace_back(put_request, curr_bytes);
    op++;
  }
  return tensorflow::Status::OK();
}

// Executions of work items.
tensorflow::Status FillTypes::RunOpImpl(int64 i, MetadataStore* store) {
  switch (fill_types_config_.specification()) {
    case FillTypesConfig::ArtifactType: {
      PutArtifactTypeRequest put_request =
          absl::get<PutArtifactTypeRequest>(work_items_[i].first);
      PutArtifactTypeResponse put_response;
      return store->PutArtifactType(put_request, &put_response);
    }
    case FillTypesConfig::ExecutionType: {
      PutExecutionTypeRequest put_request =
          absl::get<PutExecutionTypeRequest>(work_items_[i].first);
      PutExecutionTypeResponse put_response;
      return store->PutExecutionType(put_request, &put_response);
    }
    case FillTypesConfig::ContextType: {
      PutContextTypeRequest put_request =
          absl::get<PutContextTypeRequest>(work_items_[i].first);
      PutContextTypeResponse put_response;
      return store->PutContextType(put_request, &put_response);
    }
    default:
      return tensorflow::errors::InvalidArgument("Wrong specification!");
  }
  return tensorflow::errors::InvalidArgument(
      "Cannot execute the query due to wrong specification!");
}

tensorflow::Status FillTypes::TearDownImpl() {
  work_items_.clear();
  return tensorflow::Status::OK();
}

}  // namespace ml_metadata
