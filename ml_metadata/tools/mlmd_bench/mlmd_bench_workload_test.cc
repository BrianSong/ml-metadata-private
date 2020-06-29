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
#include "ml_metadata/tools/mlmd_bench/mlmd_bench_workload.h"

#include <gtest/gtest.h>

#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {

namespace {

struct WorkloadTest : public testing::Test {
  Workload<absl::variant<PutArtifactTypeRequest, PutExecutionTypeRequest,
                         PutContextTypeRequest>>* workload;
  void SetUp() override {
    workload = new Workload<
        absl::variant<PutArtifactTypeRequest, PutExecutionTypeRequest,
                      PutContextTypeRequest>>();
  }

  void TearDown() override { delete workload; }
};

// Test the constructor of the Workload class.
TEST_F(WorkloadTest, ConstructorTest) {
  EXPECT_EQ(workload->GetNumOps(), 0);
  EXPECT_FALSE(workload->GetSetUpStatus());
}

// Test the is_setup is set to true in SetUp().
TEST_F(WorkloadTest, SetUpTest) {
  ConnectionConfig mlmd_config;
  mlmd_config.mutable_fake_database();
  std::unique_ptr<MetadataStore> set_up_store;
  const MigrationOptions set_up_opts;
  CreateMetadataStore(mlmd_config, set_up_opts, &set_up_store);
  TF_EXPECT_OK(workload->SetUp(&set_up_store));
  ASSERT_TRUE(workload->GetSetUpStatus());
}

// Test the case which executing RunOp() before calling SetUp().
TEST_F(WorkloadTest, RunOpFailedPreConTest) {
  ConnectionConfig mlmd_config;
  mlmd_config.mutable_fake_database();
  std::unique_ptr<MetadataStore> store;
  const MigrationOptions opts;
  CreateMetadataStore(mlmd_config, opts, &store);
  int i = 0;
  Stats::OpStats op_stats;
  FakeClock clock;
  Watch watch(&clock);
  {
    EXPECT_EQ(workload->RunOp(i, op_stats, watch, &store).code(),
              tensorflow::error::FAILED_PRECONDITION);
  }
}

}  // namespace

}  // namespace ml_metadata
