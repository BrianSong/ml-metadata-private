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

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "ml_metadata/metadata_store/test_util.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/proto/metadata_store_service.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {

// Test fixture that uses the same data configuration for multiple following
// tests.
class FillTypesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Uses a fake in-memory SQLite database for testing.
    mlmd_config.mutable_fake_database();
    TF_ASSERT_OK(CreateMetadataStore(mlmd_config, &store));
    fill_types_config = testing::ParseTextProtoOrDie<FillTypesConfig>(
        R"(
          update: false
          specification: ARTIFACT_TYPE
          num_properties { minimum: 1 maximum: 10 }
        )");
    fill_types =
        absl::make_unique<FillTypes>(FillTypes(fill_types_config, num_ops));
  }

  std::unique_ptr<FillTypes> fill_types;
  ConnectionConfig mlmd_config;
  FillTypesConfig fill_types_config;
  std::unique_ptr<MetadataStore> store;
  int num_ops = 100;
};

// Tests the SetUpImpl() for FillTypes.
// Checks the SetUpImpl() indeed prepares a list of work items whose length is
// the same as the specified number of operations.
TEST_F(FillTypesTest, SetUpImplTest) {
  TF_ASSERT_OK(fill_types->SetUp(store.get()));
  EXPECT_EQ(num_ops, fill_types->num_ops());
}

// Tests the RunOpImpl() for insert types.
// Checks indeed all the work items have been executed and all the types have
// been inserted into the db.
TEST_F(FillTypesTest, InsertTest) {
  TF_ASSERT_OK(fill_types->SetUp(store.get()));
  for (int64 i = 0; i < fill_types->num_ops(); ++i) {
    OpStats op_stats{0, 0};
    TF_EXPECT_OK(fill_types->RunOp(i, store.get(), op_stats));
  }

  GetArtifactTypesRequest get_request;
  GetArtifactTypesResponse get_response;
  TF_ASSERT_OK(store->GetArtifactTypes(get_request, &get_response));
  EXPECT_EQ(get_response.artifact_types_size(), fill_types->num_ops());
}

}  // namespace
}  // namespace ml_metadata
