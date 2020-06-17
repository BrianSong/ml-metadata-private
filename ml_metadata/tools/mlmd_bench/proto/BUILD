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

load("//ml_metadata:ml_metadata.bzl", "ml_metadata_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

ml_metadata_proto_library(
    name = "mlmd_bench_proto",
    srcs = ["mlmd_bench.proto"],
    cc_api_version = 2,
    deps = [
        "//ml_metadata/proto:metadata_store_proto",
    ],
)