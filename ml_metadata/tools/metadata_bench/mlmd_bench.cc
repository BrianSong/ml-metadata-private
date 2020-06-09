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

#include <time.h>
#include <chrono>
#include <iostream>
#include <random>

#include "google/protobuf/text_format.h"
#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "ml_metadata/proto/metadata_bench.pb.h"
#include "ml_metadata/proto/metadata_store.pb.h"

// Number of operations for each workload
static int num_op = 10000;

namespace ml_metadata {

inline uint64_t CurrentTimeInMicroSecond() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

std::string RandomStringGenerator() {
  int n = 50;
  char alphabet[26] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                       'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                       's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

  std::string res = "";
  for (int i = 0; i < n; i++) res = res + alphabet[rand() % 26];
  return res;
}

class Stats {
 private:
  uint64_t start_;
  uint64_t finish_;
  double seconds_;
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

  void FillArtifactType(ConnectionConfig mlmd_connect_config,
                        Workload curr_workload) {
    UniformDistribution curr_num_properties_distribution =
        curr_workload.fill_types().num_properties_distribution();

    std::unique_ptr<MetadataStore> store;
    const MigrationOptions opts;
    CreateMetadataStore(mlmd_connect_config, opts, &store);

    // Random generator for uniform distribution purpose
    std::default_random_engine dre(std::chrono::steady_clock::now()
                                       .time_since_epoch()
                                       .count());  // provide seed

    Stats stats;
    stats.Start();

    for (int i = 0; i < num_op; ++i) {
      int64_t curr_bytes = 0;

      std::string type_name = RandomStringGenerator();
      curr_bytes += type_name.size();

      // Generate the number of properties according to the uniform distribution
      int min = curr_num_properties_distribution.range_begin();
      int max = curr_num_properties_distribution.range_end();
      std::uniform_int_distribution<int> uid{min, max};
      int curr_num_properties = uid(dre);

      // Generate properites of the types
      std::string property_string = "";
      for (int i = 0; i < curr_num_properties; ++i) {
        std::string curr_property_name = "property_" + std::to_string(i);
        std::string curr_property =
            "properties { key: '" + curr_property_name + "' value: STRING } ";
        property_string += curr_property;
        curr_bytes += curr_property_name.size();
      }

      // Customize the query
      std::string curr_string = R"(
			            all_fields_match: true
			            artifact_type: {
			              name: 
			              
			            }
			          )";
      curr_string.insert(94, "'" + type_name + "'");
      curr_string.insert(180, property_string);

      PutArtifactTypeRequest put_request;
      google::protobuf::TextFormat::ParseFromString(curr_string, &put_request);
      PutArtifactTypeResponse put_response;
      store->PutArtifactType(put_request, &put_response);
      stats.FinishedSingleOp();
      stats.AddBytes(curr_bytes);
    }

    stats.Stop();
    stats.Report("fill_artifact_type");
  }

  void Run(ConnectionConfig mlmd_connect_config, ThreadEnv thread_env,
           Workload curr_workload) {
    if (curr_workload.has_fill_types()) {
      if (curr_workload.fill_types().specification() ==
          FillTypes::ArtifactType) {
        // Single thread mode
        if (thread_env.thread_num() == 1) {
          FillArtifactType(mlmd_connect_config, curr_workload);
        }
      }
    }
  }
};

}  // namespace ml_metadata

int main(int argc, char** argv) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  // For now, I create the config myself and take input from the command line
  ml_metadata::MLMDBenchConfig mlmd_bench_config;
  ml_metadata::ConnectionConfig mlmd_connect_config =
      mlmd_bench_config.mlmd_connect_config();
  // ::google::protobuf::RepeatedPtrField< ml_metadata::Workload > workloads =
  // mlmd_bench_config.workloads();
  mlmd_connect_config.mutable_fake_database();

  ml_metadata::ThreadEnv thread_env = mlmd_bench_config.thread_env();
  ml_metadata::UniformDistribution uniform_distribution;

  int num_workload = 0;

  srand(time(NULL));

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
  ml_metadata::UniformDistribution* curr_num_properties_distribution =
      fill_types->mutable_num_properties_distribution();
  curr_num_properties_distribution->CopyFrom(uniform_distribution);
  num_workload++;

  ml_metadata::Benchmark benchmark;

  for (int i = 0; i < num_workload; ++i) {
    ml_metadata::Workload curr_workload = mlmd_bench_config.workloads(0);
    benchmark.Run(mlmd_connect_config, thread_env, curr_workload);
  }

  return 0;
}