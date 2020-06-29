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
#ifndef ML_METADATA_TOOLS_MLMD_BENCH_MLMD_BENCH_WATCH_H
#define ML_METADATA_TOOLS_MLMD_BENCH_MLMD_BENCH_WATCH_H

#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace ml_metadata {

// A interface that can be implemented by a real ABSLClock or a FakeClock.
class Clock {
 public:
  Clock() = default;
  virtual ~Clock() = default;

  virtual absl::Time CurrTime() = 0;
};

// A real clock used for production environment.
class ABSLClock : public Clock {
 public:
  ABSLClock() = default;
  ~ABSLClock() override = default;

  absl::Time CurrTime() override;
};

// A fake clock used for testing environment.
class FakeClock : public Clock {
 private:
  // Tell the current time. It can be set by SetTime() under testing
  // environment.
  absl::Time curr_time;

 public:
  FakeClock() = default;
  ~FakeClock() override = default;

  absl::Time CurrTime() override;

  // Set the current time for testing purposes.
  void SetTime(int second);
};

// Calculating elapsed time for certain time interval.
class Watch {
 private:
  absl::Time start_;
  absl::Time finish_;
  Clock* clock_;

 public:
  // The constructor can take in any clock depends on the current environment.
  Watch(Clock* clock);
  ~Watch() = default;

  void Start();

  void End();

  double GetElaspedTimeInMicroS();
};

}  // namespace ml_metadata

#endif  // ML_METADATA_TOOLS_MLMD_BENCH_MLMD_BENCH_WATCH_H
