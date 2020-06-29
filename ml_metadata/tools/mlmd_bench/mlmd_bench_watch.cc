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

namespace ml_metadata {

absl::Time ABSLClock::CurrTime() { return absl::Now(); }

absl::Time FakeClock::CurrTime() { return curr_time; }

void FakeClock::SetTime(int second) {
  auto t1 = std::chrono::system_clock::from_time_t(second);
  curr_time = absl::FromChrono(t1);
}

Watch::Watch(Clock* clock) : clock_(clock) {}

void Watch::Start() { start_ = clock_->CurrTime(); }

void Watch::End() { finish_ = clock_->CurrTime(); }

double Watch::GetElaspedTimeInMicroS() {
  return (finish_ - start_) / (absl::Microseconds(1));
}

}  // namespace ml_metadata
