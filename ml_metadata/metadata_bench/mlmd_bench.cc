/* Copyright 2019 Google LLC

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

#include <iostream>
#include <chrono>
// #include <unistd.h>


#include "ml_metadata/proto/metadata_bench.pb.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "tensorflow/core/lib/core/status.h"

// Number of operations for each workload
static int num_op = 100000;

namespace mlmd_bench {

	inline uint64_t CurrentTimeInMicroSecond() {
	    return std::chrono::duration_cast<std::chrono::microseconds>
	               (std::chrono::high_resolution_clock::now().time_since_epoch()).count();
	}

	class Stats {
	 private:
	  uint64_t start_;
	  uint64_t finish_;
	  uint64_t seconds_;
	  int done_;
	  int next_report_;
	  int64_t bytes_;

	 public:
	  Stats() { Start(); }

	  void Start() {
	    next_report_ = 100;
	    done_ = 0;
	    bytes_ = 0;
	    seconds_ = 0;
	    start_ = finish_ = CurrentTimeInMicroSecond();
	  }

	  void Merge(const Stats& other) {
	    done_ += other.done_;
	    bytes_ += other.bytes_;
	    seconds_ += other.seconds_;
	    if (other.start_ < start_) start_ = other.start_;
	    if (other.finish_ > finish_) finish_ = other.finish_;
	  }

	  void Stop() {
	    finish_ = CurrentTimeInMicroSecond();
	    seconds_ = (finish_ - start_) * 1e-6;
	  }

	  void FinishedSingleOp() {
	    done_++;
	    if (done_ >= next_report_) {
	      if (next_report_ < 1000)
	        next_report_ += 100;
	      else if (next_report_ < 5000)
	        next_report_ += 500;
	      else if (next_report_ < 10000)
	        next_report_ += 1000;
	      else if (next_report_ < 50000)
	        next_report_ += 5000;
	      else if (next_report_ < 100000)
	        next_report_ += 10000;
	      else if (next_report_ < 500000)
	        next_report_ += 50000;
	      else
	        next_report_ += 100000;
	      std::fprintf(stderr, "... finished %d ops%30s\r", done_, "");
	      std::fflush(stderr);
	    }
	  }

	  void AddBytes(int64_t n) { bytes_ += n; }

	  void Report(std::string specification) {
	    // Pretend at least one op was done in case we are running a specification
	    // that does not call FinishedSingleOp().
	    if (done_ < 1) done_ = 1;

	    std::string extra;
	    if (bytes_ > 0) {
	      // Rate is computed on actual elapsed time, not the sum of per-thread
	      // elapsed times.
	      double elapsed = (finish_ - start_) * 1e-6;
	      char rate[100];
	      std::snprintf(rate, sizeof(rate), "%6.1f MB/s",
	                    (bytes_ / 1048576.0) / elapsed);
	      extra = rate;
	    }

	    std::fprintf(stdout, "%-12s : %11.3f micros/op;%s%s\n",
	                 specification.c_str(), seconds_ * 1e6 / done_,
	                 (extra.empty() ? "" : " "), extra.c_str());
	    std::fflush(stdout);
	  }
	};


	class Benchmark {
	 public:
	 	Benchmark() {}

	 	void Run(ml_metadata::ConnectionConfig mlmd_config, ml_metadata::ThreadEnv thread_env, ml_metadata::Workload curr_workload) {
	 		if (curr_workload.has_fill_types()) {
	 			if (curr_workload.fill_types().specification() == ml_metadata::FillTypes::ArtifactType) {
	 				// method = &Benchmark::FillArtifactType;
	 			}
	 		}
	 	}
	};

} // namespace mlmd_bench
 
int main(int argc, char** argv) {
	// Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
	GOOGLE_PROTOBUF_VERIFY_VERSION;

	ml_metadata::MLMDBenchConfig mlmd_bench_config;
	ml_metadata::ConnectionConfig mlmd_config = mlmd_bench_config.mlmd_config();
	// ::google::protobuf::RepeatedPtrField< ml_metadata::Workload > workloads = mlmd_bench_config.workloads();
	ml_metadata::ThreadEnv thread_env = mlmd_bench_config.thread_env();
	ml_metadata::UniformDistribution uniform_distribution;

	int num_workload = 0;


	for (int i = 1; i < argc; ++i) {
		char junk;
		int n;
		if (sscanf(argv[i], "--num_thread=%d%c", &n, &junk) == 1) {
			thread_env.set_thread_num(n);
		} else if (sscanf(argv[i], "--UD_begin=%d%c", &n, &junk) == 1) {
			uniform_distribution.set_range_begin(n);
		} else if (sscanf(argv[i], "--UD_end=%d%c", &n, &junk) == 1) {
			uniform_distribution.set_range_end(n);
		}
	}


	// For now, we are only testing against fill_artifact_type specification
	ml_metadata::Workload* new_workload = mlmd_bench_config.add_workloads();
	ml_metadata::FillTypes* fill_types = new_workload->mutable_fill_types();
	fill_types->set_specification(ml_metadata::FillTypes::ArtifactType);
	ml_metadata::UniformDistribution* curr_num_properties_distribution = fill_types->mutable_num_properties_distribution();
	curr_num_properties_distribution->CopyFrom(uniform_distribution);
	num_workload ++;

	mlmd_bench::Benchmark benchmark;

	for (int i = 0; i < num_workload; ++i) {
		ml_metadata::Workload curr_workload = mlmd_bench_config.workloads(0);
		benchmark.Run(mlmd_config, thread_env, curr_workload);
	}

	return 0;
}