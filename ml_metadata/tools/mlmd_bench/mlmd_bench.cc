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
#include <time.h>

#include <iostream>
#include <random>
#include <vector>

#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "google/protobuf/text_format.h"

#include "ml_metadata/metadata_store/metadata_store.h"
#include "ml_metadata/metadata_store/metadata_store_factory.h"
#include "ml_metadata/proto/metadata_store.pb.h"
#include "ml_metadata/tools/mlmd_bench/proto/mlmd_bench.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace ml_metadata {

namespace {

// Initialize the mlmd_bench_config from the user input in command line
void InitMLMDBenchConfigFromCommandLineArgs(
    int argc, char** argv, ml_metadata::MLMDBenchConfig& mlmd_bench_config) {
  ml_metadata::ConnectionConfig mlmd_connect_config;
  ml_metadata::ThreadEnvConfig thread_env_config;
  ml_metadata::UniformDistribution uniform_distribution;
  std::string specification;

  for (int i = 1; i < argc; ++i) {
    char junk;
    int n;
    char s[20];
    if (sscanf(argv[i], "--a=%d%c", &n, &junk) == 1) {
      uniform_distribution.set_a(n);
    } else if (sscanf(argv[i], "--b=%d%c", &n, &junk) == 1) {
      uniform_distribution.set_b(n);
    } else if (sscanf(argv[i], "--type=%s%c", s, &junk) == 1) {
      specification = s;
    } else if (sscanf(argv[i], "--num_ops=%d%c", &n, &junk) == 1) {
      thread_env_config.set_num_ops(n);
    } else if (sscanf(argv[i], "--num_threads=%d%c", &n, &junk) == 1) {
      thread_env_config.set_num_threads(n);
    }
  }

  // Update the connection and thread environment config. according to the
  // command line input
  mlmd_connect_config.mutable_fake_database();
  mlmd_bench_config.mutable_mlmd_connect_config()->CopyFrom(
      mlmd_connect_config);
  mlmd_bench_config.mutable_thread_env_config()->CopyFrom(thread_env_config);

  // Create the workload config. according to the user input
  // For now, we only test against the FillTypes workload
  ml_metadata::WorkloadConfig* new_workload_config =
      mlmd_bench_config.add_workload_configs();
  ml_metadata::FillTypesConfig* fill_types =
      new_workload_config->mutable_fill_types();
  if (specification == "artifact") {
    fill_types->set_specification(ml_metadata::FillTypesConfig::ArtifactType);
  } else if (specification == "execution") {
    fill_types->set_specification(ml_metadata::FillTypesConfig::ExecutionType);
  } else if (specification == "context") {
    fill_types->set_specification(ml_metadata::FillTypesConfig::ContextType);
  } else {
    std::cout << "Input type error!" << std::endl;
    return;
  }
  fill_types->mutable_num_properties()->CopyFrom(uniform_distribution);
}

// Generate random (can be approximated to be unique string)
// for fulfilling the unique type name constrict
std::string RandomStringGenerator() {
  int n = 50;
  std::string res;
  for (int i = 0; i < n; i++) {
    char random_char = 'a' + rand() % 26;
    res = absl::StrCat(res, std::string(1, random_char));
  }
  return res;
}

// TODO(briansong): move this Stats class to .h/.cc
// Stats class is design specifically to record the statics along the
// benchmarking process
class Stats {
 private:
  double elapsed_micros_;
  int done_;
  int next_report_;
  int64_t bytes_;

 public:
  Stats() { Start(); }

  // Start the stats instance
  void Start() {
    next_report_ = 100;
    done_ = 0;
    bytes_ = 0;
    elapsed_micros_ = 0;
  }

  // Being called at the end of each operation to record the elapsed time for
  // each operation and the number of total operations performed by each thread
  void FinishedSingleOp(const absl::Time& start_time,
                        const absl::Time& end_time) {
    done_++;
    elapsed_micros_ += (end_time - start_time) / absl::Microseconds(1);
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

  // Being called at the end of each operation to add the transferred bytes of
  // the current operation to the total bytes being transferred for each thread
  void AddBytes(int64_t n) { bytes_ += n; }

  // Report the metrics of interests: microsecond per operation and total bytes
  // per seconds for the current workload
  void Report(const std::string& specification) {
    // Pretend at least one op was done in case we are running a specification
    // that does not call FinishedSingleOp().
    if (done_ < 1) done_ = 1;

    std::string extra;
    if (bytes_ > 0) {
      // Rate is computed on actual elapsed time, not the sum of per-thread
      // elapsed times.
      char rate[100];
      std::snprintf(rate, sizeof(rate), "%6.1f MB/s",
                    (bytes_ / 1048576.0) / (elapsed_micros_ * 1e-6));
      extra = rate;
    }

    std::fprintf(stdout, "%-12s : %11.3f micros/op;%s%s\n",
                 specification.c_str(), elapsed_micros_ / done_,
                 (extra.empty() ? "" : " "), extra.c_str());
    std::fflush(stdout);
  }
};

// TODO(briansong): move this Workload and its children classes to .h/.cc
// The generalized workload class
class Workload {
 public:
  Workload(){};
  virtual std::pair<std::string, int> SetUpOp() {
    return std::make_pair("", 0);
  };
  virtual bool Op(std::string query_string) { return true; };
  virtual void TearDownOp(){};
  virtual std::string GetSpecification() { return ""; };
};

// FillTypes class is used to specify filling artifact_type, exectition_type and
// context_type to the database
class FillTypes : public Workload {
 private:
  std::unique_ptr<MetadataStore>* curr_store_ptr;
  WorkloadConfig curr_workload_config;
  std::string specification;
  std::minstd_rand0 gen;

 public:
  FillTypes(std::unique_ptr<MetadataStore>* store_ptr,
            const WorkloadConfig& workload_config) {
    curr_store_ptr = store_ptr;
    curr_workload_config = workload_config;
    switch (curr_workload_config.fill_types().specification()) {
      case (0):
        specification = "artifact_type";
        break;
      case (1): {
        specification = "execution_type";
        break;
      }
      case (2):
        specification = "context_type";
        break;
      default:
        std::cout << "Specification Error!" << std::endl;
        break;
    }
    gen.seed(absl::ToUnixMillis(absl::Now()));
  }

  // Set up operation for FillTypes workload, it will not be included into the
  // performance evaluation
  std::pair<std::string, int> SetUpOp() {
    int curr_bytes = 0;
    std::string query_string;
    UniformDistribution num_properties =
        curr_workload_config.fill_types().num_properties();
    std::string type_name = RandomStringGenerator();
    curr_bytes += type_name.size();

    // Generate the number of properties according to the uniform distribution
    int min = num_properties.a();
    int max = num_properties.b();
    std::uniform_int_distribution<int> uid{min, max};
    int curr_num_properties = uid(gen);

    // Generate properties of the types
    std::string property_string;
    for (int i = 0; i < curr_num_properties; ++i) {
      std::string curr_property_name = "property_" + std::to_string(i);
      std::string curr_property =
          "properties { key: '" + curr_property_name + "' value: STRING } ";
      property_string += curr_property;
      curr_bytes += curr_property_name.size();
    }

    // Customize the query
    query_string = R"(
                  all_fields_match: true
                  : {
                    name: 
                    
                  }
                )";
    query_string.insert(60, specification);
    query_string.insert(104, "'" + type_name + "'");
    query_string.insert(180, property_string);
    return std::make_pair(query_string, curr_bytes);
  }

  // The real operation being performed against the database
  // The elapsed time for this operation will be tracked
  bool Op(std::string query_string) {
    if (specification == "artifact_type") {
      PutArtifactTypeRequest put_request;
      google::protobuf::TextFormat::ParseFromString(query_string, &put_request);
      PutArtifactTypeResponse put_response;
      return ((*curr_store_ptr)->PutArtifactType(put_request, &put_response))
          .ok();
    } else if (specification == "execution_type") {
      PutExecutionTypeRequest put_request;
      google::protobuf::TextFormat::ParseFromString(query_string, &put_request);
      PutExecutionTypeResponse put_response;
      return ((*curr_store_ptr)->PutExecutionType(put_request, &put_response))
          .ok();
    } else if (specification == "context_type") {
      PutContextTypeRequest put_request;
      google::protobuf::TextFormat::ParseFromString(query_string, &put_request);
      PutContextTypeResponse put_response;
      return ((*curr_store_ptr)->PutContextType(put_request, &put_response))
          .ok();
    } else {
      std::cout << "Cannot execute the query due to wrong specification!"
                << std::endl;
      return false;
    }
  }

  // Get the specification (artifact_type, exectition_type or context_type) of
  // the FillTypes
  std::string GetSpecification() { return specification; }
};

}  // namespace

// TODO(briansong): move this Benchmark class to .h/.cc
// Benchmark class contains a list of workloads to be executed by ThreadRunner
// It takes the workload configurations and generate executable workloads based
// on that
class Benchmark {
 private:
  std::unique_ptr<MetadataStore> store;
  Workload curr_workload;

 public:
  // TODO(briansong): cannot pass workloads out using get function since unique
  // pointer cannot be copied, is there a better way other than setting it to
  // public scope?
  std::vector<std::unique_ptr<Workload>> workloads;
  Benchmark() {}
  Benchmark(const MLMDBenchConfig& mlmd_bench_config) {
    const MigrationOptions opts;
    CreateMetadataStore(mlmd_bench_config.mlmd_connect_config(), opts, &store);
    for (const WorkloadConfig& workload_config :
         mlmd_bench_config.workload_configs()) {
      CreateWorkloadAndSave(workload_config);
    }
  }

  // Take an array of workload configurations and generate corresponding
  // workloads
  void CreateWorkloadAndSave(const WorkloadConfig& workload_config) {
    switch (workload_config.workload_case()) {
      case (1):
        std::cout << "Not implement InitStore yet" << std::endl;
        break;
      case (2): {
        // TODO(briansong): Use unique pointer here because otherwise the sub
        // workload will be destroyed outside the current scope(cannot use raw
        // pointer since it will point to leased space), we can not pass out by
        // value since then, it will be declared upon Workload class and lose
        // the access to the override functions of subclass. Is it a good
        // approach?
        std::unique_ptr<FillTypes> p(new FillTypes(&store, workload_config));
        workloads.push_back(std::move(p));
        break;
      }
      case (3):
        std::cout << "Not implement ReadTypes yet" << std::endl;
        break;
      default:
        std::cout << "Invalid workload!" << std::endl;
    }
  }
};

// TODO(briansong): move this ThreadRunner class to .h/.cc
// The ThreadRunner class is the execution component of the mlmd_bench
// It takes the benchmark and run the workloads specified inside
class ThreadRunner {
 private:
  int num_threads;
  int num_op;
  std::mutex mtx;

 public:
  ThreadRunner(const ThreadEnvConfig& thread_env_config) {
    num_threads = thread_env_config.num_threads();
    num_op = thread_env_config.num_ops();
  }

  // Execution unit of mlmd_bench
  void Run(Benchmark& benchmark) {
    int op_per_thread = num_op / num_threads;
    for (auto& workload : benchmark.workloads) {
      Stats stats;
      stats.Start();
      for (int i = 0; i < op_per_thread; ++i) {
        std::pair<std::string, int> res = workload->SetUpOp();
        std::string query_string = res.first;
        int curr_bytes = res.second;
        absl::Time start_time = absl::Now();
        if (workload->Op(query_string)) {
          absl::Time end_time = absl::Now();
          stats.FinishedSingleOp(start_time, end_time);
          stats.AddBytes(curr_bytes);
        } else {
          std::cout << "Fail execution!" << std::endl;
          return;
        }
      }
      stats.Report("fill_" + workload->GetSpecification());
    }
  }
};

}  // namespace ml_metadata

int main(int argc, char** argv) {
  // Create the MLMDBenchConfig and take input from the command line
  ml_metadata::MLMDBenchConfig mlmd_bench_config;
  ml_metadata::InitMLMDBenchConfigFromCommandLineArgs(argc, argv,
                                                      mlmd_bench_config);

  srand(time(NULL));
  ml_metadata::Benchmark benchmark(mlmd_bench_config);
  ml_metadata::ThreadRunner runner(mlmd_bench_config.thread_env_config());
  runner.Run(benchmark);

  return 0;
}