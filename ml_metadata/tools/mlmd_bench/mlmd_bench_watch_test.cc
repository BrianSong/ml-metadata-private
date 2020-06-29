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
#include "ml_metadata/tools/mlmd_bench/mlmd_bench_watch.h"

#include <gtest/gtest.h>

namespace ml_metadata {

// Test the elapsed time functionary for Watch class.
TEST(WatchTest, TestElapsedTime) {
  // Use a FakeClock to avoid flaky behavior of ABSLClock
  FakeClock clock;
  Watch watch(&clock);
  clock.SetTime(3);
  watch.Start();
  clock.SetTime(13);
  watch.End();
  ASSERT_EQ(watch.GetElaspedTimeInMicroS(), 10 * 1e6);
}

}  // namespace ml_metadata
