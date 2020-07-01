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
#include "ml_metadata/tools/mlmd_bench/workload.h"

#include <gtest/gtest.h>

#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace ml_metadata {
namespace {

class FakeWorkload : public Workload<std::string> {
  tensorflow::Status SetUpImpl(MetadataStore* store_ptr) {
    work_items_bytes_.push_back(8888);
    return tensorflow::Status::OK();
  }

  tensorflow::Status RunOpImpl(int i, MetadataStore* store_ptr) {
    return tensorflow::Status::OK();
  }

  tensorflow::Status TearDownImpl() { return tensorflow::Status::OK(); }
};

// Test successfulness when executing in the right sequence.
TEST(WorkloadTest, RunInRightSequenceTest) {
  FakeWorkload workload;
  ConnectionConfig mlmd_config;
  mlmd_config.mutable_fake_database();

  std::unique_ptr<MetadataStore> store;
  const MigrationOptions opts;
  CreateMetadataStore(mlmd_config, opts, &store);

  int i = 0;
  OpStats op_stats;
  FakeClock clock;
  Watch watch(&clock);

  TF_ASSERT_OK(workload.SetUp(store.get()));
  TF_EXPECT_OK(workload.RunOp(i, watch, store.get(), op_stats));
  TF_EXPECT_OK(workload.TearDown());
}

// Test the cases when executing RunOp() / TearDown() before calling SetUp().
TEST(WorkloadTest, FailedPreconditionTest) {
  FakeWorkload workload;
  ConnectionConfig mlmd_config;
  mlmd_config.mutable_fake_database();

  std::unique_ptr<MetadataStore> store;
  const MigrationOptions opts;
  CreateMetadataStore(mlmd_config, opts, &store);

  int i = 0;
  OpStats op_stats;
  FakeClock clock;
  Watch watch(&clock);
  EXPECT_EQ(workload.RunOp(i, watch, store.get(), op_stats).code(),
            tensorflow::error::FAILED_PRECONDITION);
  EXPECT_EQ(workload.TearDown().code(), tensorflow::error::FAILED_PRECONDITION);
}

}  // namespace
}  // namespace ml_metadata
