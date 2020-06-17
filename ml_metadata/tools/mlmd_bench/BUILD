# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# GOOGLE BEGIN-INTERNAL
# TODO(b/120486527): Make this a function of the original BUILD file.
# This is a manually written bazel file. Each protobufs rule internally havs to be
# rewritten as two rules externally. tensorflow-data-validation solves this via
# ml_metadata.bzl. However, I haven't got that working here. When I do, I will
# drop this version of the file.
# GOOGLE END-INTERNAL

cc_binary(
    name = "mlmd_bench",
    srcs = ["mlmd_bench.cc"],
    deps = [
    	"@com_google_protobuf//:protobuf",
    	"@com_google_absl//absl/time",
    	"@com_google_absl//absl/strings",
    	"//ml_metadata/proto:metadata_store_proto",
    	"//ml_metadata/metadata_store:metadata_store",
    	"//ml_metadata/metadata_store:metadata_store_factory",
    	"//ml_metadata/tools/mlmd_bench/proto:mlmd_bench_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ]
)