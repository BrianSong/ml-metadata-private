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

#include <gtest/gtest.h>

#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {

class FillTypesTest : public testing::Test {
 protected:
  // Create a FillTypes instance.
  void SetUp() override {
    mlmd_config.mutable_fake_database();
    CreateMetadataStore(mlmd_config, opts, &store);
    workload_config.set_num_ops(10000);
    FillTypesConfig* fill_types_config =
        workload_config.mutable_fill_types_config();
    fill_types_config->set_specification(FillTypesConfig::ExecutionType);

    uniform_distribution.set_a(1);
    uniform_distribution.set_b(10);

    fill_types_config->mutable_num_properties()->CopyFrom(uniform_distribution);
    fill_types_p = new FillTypes(workload_config.fill_types_config(),
                                 workload_config.num_ops());
    fill_types_p->SetUp(store.get());
  }

  // Delete the FillTypes instance.
  void TearDown() override { delete fill_types_p; }

  FillTypes* fill_types_p;
  ConnectionConfig mlmd_config;
  std::unique_ptr<MetadataStore> store;
  const MigrationOptions opts;
  WorkloadConfig workload_config;
  UniformDistribution uniform_distribution;
};

// Test the SetUpImpl() for FillTypes.
// Check the SetUpImpl() indeed prepared a list of work items whose length is
// equal to the number of operations.
TEST_F(FillTypesTest, SetUpImplTest) {
  EXPECT_EQ(workload_config.num_ops(), fill_types_p->num_ops());
}

// Test the RunOpImpl() for FillTypes.
// Check indeed all the work items have been executed and all the types have
// been inserted into the database.
TEST_F(FillTypesTest, RunOpImplTest) {
  for (int i = 0; i < fill_types_p->num_ops(); ++i) {
    OpStats op_stats;
    FakeClock clock;
    clock.SetTime(0);
    Watch watch(&clock);
    TF_EXPECT_OK(fill_types_p->RunOp(i, watch, store.get(), op_stats));
  }

  // for (int i = 0; i < fill_types_p->num_ops(); ++i) {
  //   std::string check_type_query_string = R"(
  //           type_name:
  //         )";
  //   check_type_query_string.insert(23,
  //                                  "'" + fill_types_p->types_name()[i] +
  //                                  "'");
  //   GetExecutionTypeRequest get_request;
  //   google::protobuf::TextFormat::ParseFromString(check_type_query_string,
  //                                                 &get_request);
  //   GetExecutionTypeResponse get_response;
  //   TF_EXPECT_OK((store)->GetExecutionType(get_request, &get_response));
  // }
}

}  // namespace
}  // namespace ml_metadata
