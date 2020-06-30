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

// Generate random (will be check for uniqueness)
// for fulfilling the unique type name constrict.
std::string RandomStringGenerator() {
  int n = 50;
  std::string res;
  for (int i = 0; i < n; i++) {
    char random_char = 'a' + rand() % 26;
    res = absl::StrCat(res, std::string(1, random_char));
  }
  return res;
}

}  // namespace

// FillTypes constructor for setting up its configurations.
FillTypes::FillTypes(const WorkloadConfig& workload_config)
    : workload_config_(workload_config) {
  num_ops_ = workload_config.num_ops();
  // Each FillTypes will have its specification to indicate which type to
  // operate on.
  switch (workload_config_.fill_types_config().specification()) {
    case (0):
      specification_ = "artifact_type";
      break;
    case (1): {
      specification_ = "execution_type";
      break;
    }
    case (2):
      specification_ = "context_type";
      break;
    default:
      std::cout << "Specification Error!" << std::endl;
      break;
  }
  // The seed for the random generator will be the time when the FillTypes is
  // created.
  gen_.seed(absl::ToUnixMillis(absl::Now()));
}

tensorflow::Status FillTypes::SetUpImpl(
    std::unique_ptr<MetadataStore>*& set_up_store_ptr) {
  std::fprintf(stderr, "Setting up ...");
  std::fflush(stderr);
  int op = 0;
  while (op < num_ops_) {
    // Calculate the transferred bytes for executing each work item.
    int curr_bytes = 0;
    // Real query string for executing the workload.
    std::string query_string;
    // Query string used to check if the generated type name is unique and the
    // type is not inserted into the database before.
    std::string check_type_query_string;
    // Uniform distribution describing the number of properties for each
    // generated types.
    UniformDistribution num_properties =
        workload_config_.fill_types_config().num_properties();
    std::string type_name = RandomStringGenerator();
    curr_bytes += type_name.size();

    // Generate the number of properties according to the uniform distribution.
    int min = num_properties.a();
    int max = num_properties.b();
    std::uniform_int_distribution<int> uid{min, max};
    int curr_num_properties = uid(gen_);

    // Generate properties of the types.
    std::string property_string;
    for (int i = 0; i < curr_num_properties; ++i) {
      std::string curr_property_name = "property_" + std::to_string(i);
      std::string curr_property =
          "properties { key: '" + curr_property_name + "' value: STRING } ";
      property_string += curr_property;
      curr_bytes += curr_property_name.size();
    }

    // Generate the query string for checking whether the type has been inserted
    // into the database before.
    check_type_query_string = R"(
            type_name:
          )";
    check_type_query_string.insert(23, "'" + type_name + "'");

    // Generate the executed query string for RunOp().
    query_string = R"(
                  all_fields_match: true
                  : {
                    name: 
                    
                  }
                )";
    query_string.insert(60, specification_);
    query_string.insert(104, "'" + type_name + "'");
    query_string.insert(180, property_string);

    // Update the types_name_, setup_work_items_ and setup_work_items_bytes_
    // according the specification.
    if (specification_ == "artifact_type") {
      GetArtifactTypeRequest get_request;
      google::protobuf::TextFormat::ParseFromString(check_type_query_string,
                                                    &get_request);
      GetArtifactTypeResponse get_response;
      // If the type has been existed inside the database(the type name
      // generated this time is not unique), we skip the current operation and
      // regenerate the type name.
      if ((*set_up_store_ptr)
              ->GetArtifactType(get_request, &get_response)
              .ok()) {
        continue;
      }
      PutArtifactTypeRequest put_request;
      google::protobuf::TextFormat::ParseFromString(query_string, &put_request);
      types_name_.push_back(type_name);
      setup_work_items_.push_back(put_request);
      setup_work_items_bytes_.push_back(curr_bytes);
    } else if (specification_ == "execution_type") {
      GetExecutionTypeRequest get_request;
      google::protobuf::TextFormat::ParseFromString(check_type_query_string,
                                                    &get_request);
      GetExecutionTypeResponse get_response;
      if ((*set_up_store_ptr)
              ->GetExecutionType(get_request, &get_response)
              .ok()) {
        continue;
      }
      PutExecutionTypeRequest put_request;
      google::protobuf::TextFormat::ParseFromString(query_string, &put_request);
      types_name_.push_back(type_name);
      setup_work_items_.push_back(put_request);
      setup_work_items_bytes_.push_back(curr_bytes);
    } else if (specification_ == "context_type") {
      GetContextTypeRequest get_request;
      google::protobuf::TextFormat::ParseFromString(check_type_query_string,
                                                    &get_request);
      GetContextTypeResponse get_response;
      if ((*set_up_store_ptr)
              ->GetContextType(get_request, &get_response)
              .ok()) {
        continue;
      }
      PutContextTypeRequest put_request;
      google::protobuf::TextFormat::ParseFromString(query_string, &put_request);
      types_name_.push_back(type_name);
      setup_work_items_.push_back(put_request);
      setup_work_items_bytes_.push_back(curr_bytes);
    }
    op++;
  }
  return tensorflow::Status::OK();
}

// Executing the work item prepared in SetUpImpl().
tensorflow::Status FillTypes::RunOpImpl(
    int i, std::unique_ptr<MetadataStore>*& store_ptr) {
  if (specification_ == "artifact_type") {
    PutArtifactTypeRequest put_request =
        absl::get<PutArtifactTypeRequest>(setup_work_items_[i]);
    PutArtifactTypeResponse put_response;
    return (*store_ptr)->PutArtifactType(put_request, &put_response);
  } else if (specification_ == "execution_type") {
    PutExecutionTypeRequest put_request =
        absl::get<PutExecutionTypeRequest>(setup_work_items_[i]);
    PutExecutionTypeResponse put_response;
    return (*store_ptr)->PutExecutionType(put_request, &put_response);
  } else if (specification_ == "context_type") {
    PutContextTypeRequest put_request =
        absl::get<PutContextTypeRequest>(setup_work_items_[i]);
    PutContextTypeResponse put_response;
    return (*store_ptr)->PutContextType(put_request, &put_response);
  } else {
    return tensorflow::errors::InvalidArgument(
        "Cannot execute the query due to wrong specification!");
  }
}

tensorflow::Status FillTypes::TearDownImpl(Stats& thread_stats) {
  setup_work_items_.clear();
  setup_work_items_bytes_.clear();
  types_name_.clear();
  return tensorflow::Status::OK();
}

std::vector<FillTypeWorkItemType> FillTypes::GetWorkItem() {
  return setup_work_items_;
}

std::vector<int> FillTypes::GetWorkItemBytes() {
  return setup_work_items_bytes_;
}

std::vector<std::string> FillTypes::GetTypesName() { return types_name_; }

}  // namespace ml_metadata
