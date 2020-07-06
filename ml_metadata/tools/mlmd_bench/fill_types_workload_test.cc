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
          specification: ArtifactType
          num_properties { a: 1 b: 10 }
        )");
    fill_types =
        absl::make_unique<FillTypes>(FillTypes(fill_types_config, 100));
  }

  std::unique_ptr<FillTypes> fill_types;
  ConnectionConfig mlmd_config;
  FillTypesConfig fill_types_config;
  std::unique_ptr<MetadataStore> store;
};

// Tests the SetUpImpl() for FillTypes.
// Checks the SetUpImpl() indeed prepares a list of work items whose length is
// the same as the specified number of operations.
TEST_F(FillTypesTest, SetUpImplTest) {
  TF_ASSERT_OK(fill_types->SetUp(store.get()));
  EXPECT_EQ(100, fill_types->num_ops());
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

// Tests the invalid num_ops cases.
// The num_ops should be smaller or equal to the number of existed types in db.
TEST_F(FillTypesTest, MoreTypesToUpdateThanExistTest) {
  // First runs the insert operations to ensure the db has some types inside.
  TF_ASSERT_OK(fill_types->SetUp(store.get()));
  for (int64 i = 0; i < fill_types->num_ops(); ++i) {
    OpStats op_stats;
    TF_EXPECT_OK(fill_types->RunOp(i, store.get(), op_stats));
  }

  std::unique_ptr<FillTypes> fill_types_update;
  // Updates configuration where update is set to true.
  FillTypesConfig fill_types_update_config =
      testing::ParseTextProtoOrDie<FillTypesConfig>(
          R"(
            update: true
            specification: ArtifactType
            num_properties { a: 1 b: 10 }
          )");
  fill_types_update =
      absl::make_unique<FillTypes>(FillTypes(fill_types_update_config, 101));
  // Since 101 > 100, the SetUpImpl() inside SetUp() should return Invalid
  // Argument error.
  EXPECT_EQ(fill_types_update->SetUp(store.get()).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

// Tests the RunOpImpl() for update types.
TEST_F(FillTypesTest, UpdateTest) {
  TF_ASSERT_OK(fill_types->SetUp(store.get()));
  for (int64 i = 0; i < fill_types->num_ops(); ++i) {
    OpStats op_stats;
    TF_EXPECT_OK(fill_types->RunOp(i, store.get(), op_stats));
  }

  // Gets the get_response_before_update for later comparison.
  GetArtifactTypesRequest get_request_before_update;
  GetArtifactTypesResponse get_response_before_update;
  TF_ASSERT_OK(store->GetArtifactTypes(get_request_before_update,
                                       &get_response_before_update));

  std::unique_ptr<FillTypes> fill_types_update;
  FillTypesConfig fill_types_update_config =
      testing::ParseTextProtoOrDie<FillTypesConfig>(
          R"(
            update: true
            specification: ArtifactType
            num_properties { a: 1 b: 10 }
          )");
  fill_types_update =
      absl::make_unique<FillTypes>(FillTypes(fill_types_update_config, 100));
  TF_ASSERT_OK(fill_types_update->SetUp(store.get()));
  for (int64 i = 0; i < fill_types_update->num_ops(); ++i) {
    OpStats op_stats;
    TF_EXPECT_OK(fill_types_update->RunOp(i, store.get(), op_stats));
  }

  // Gets the get_response_after_update for later comparison.
  GetArtifactTypesRequest get_request_after_update;
  GetArtifactTypesResponse get_response_after_update;
  TF_ASSERT_OK(store->GetArtifactTypes(get_request_after_update,
                                       &get_response_after_update));

  // If the updates are working properly, the type name should remain the same
  // even after the updates. On the other hand, the properties size for each
  // type should be greater than before since some new fields have been added to
  // each type.
  for (int64 i = 0; i < fill_types_update->num_ops(); ++i) {
    EXPECT_STREQ(get_response_before_update.artifact_types()[i].name().c_str(),
                 get_response_after_update.artifact_types()[i].name().c_str());
    EXPECT_LT(
        get_response_before_update.artifact_types()[i].properties().size(),
        get_response_after_update.artifact_types()[i].properties().size());
  }
}

}  // namespace
}  // namespace ml_metadata
