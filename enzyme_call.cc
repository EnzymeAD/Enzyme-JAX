//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/RWMutex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"

#include "pybind11/pybind11.h"

#include "clang_compile.h"

namespace {
class CpuKernel {
 public:
  CpuKernel(int64_t identifier, llvm::StringRef source,
            llvm::ArrayRef<llvm::SmallVector<int64_t>> out_shapes, bool dump_ir)
      : identifier(identifier) {
    if (dump_ir) {
      llvm::errs() << "constructing forward for " << source << "\n";
    }
    this->out_shapes.assign(out_shapes.begin(), out_shapes.end());
  }

  static int64_t create(llvm::StringRef source,
                        llvm::ArrayRef<llvm::SmallVector<int64_t>> out_shapes,
                        bool dump_ir) {
    llvm::sys::SmartScopedWriter<true> lock(kernel_mutex);
    int64_t identifier = last_identifier++;

    std::string output;
    int64_t status = GetLLVMFromJob("/enzyme_call/source.cpp", source.str(), output);
    if (status != 0)
      throw pybind11::value_error("failed to compile C++");

    // TODO: replace this with something more useful
    llvm::errs() << "compiled c++ to: " << output << "\n";

    kernels.try_emplace(
        identifier,
        std::make_unique<CpuKernel>(identifier, source, out_shapes, dump_ir));
    return identifier;
  }

  static CpuKernel *get(int64_t identifier) {
    llvm::sys::SmartScopedReader<true> lock(kernel_mutex);
    auto it = kernels.find(identifier);
    if (it == kernels.end()) return nullptr;
    return it->getSecond().get();
  }

  void call(void *out, void **ins) const {
    void **outs = out_shapes.size() > 1 ? reinterpret_cast<void **>(out) : &out;

    for (int i = 0, e = out_shapes.size(); i < e; ++i) {
      int64_t num_elements =
          std::accumulate(out_shapes[i].begin(), out_shapes[i].end(), 1,
                          std::multiplies<int64_t>());
      for (int j = 0; j < num_elements; ++j) {
        if (i == 0) {
          reinterpret_cast<float *>(outs[i])[j] =
              static_cast<float>(identifier);
        } else {
          reinterpret_cast<float *>(outs[i])[j] = 41.0f + i;
        }
      }
    }
  }

 private:
  static llvm::DenseMap<int64_t, std::unique_ptr<CpuKernel>> kernels;
  static int64_t last_identifier;
  static llvm::sys::SmartRWMutex<true> kernel_mutex;

  int64_t identifier;
  llvm::SmallVector<llvm::SmallVector<int64_t>> out_shapes;
};

llvm::DenseMap<int64_t, std::unique_ptr<CpuKernel>>
    CpuKernel::kernels;
int64_t CpuKernel::last_identifier = 1;
llvm::sys::SmartRWMutex<true> CpuKernel::kernel_mutex;
}  // namespace

void CpuCallback(void *out, void **ins) {
  int64_t identifier = *reinterpret_cast<int64_t *>(ins[0]);
  CpuKernel *kernel = CpuKernel::get(identifier);
  if (!kernel) {
    // TODO: find a way to fail more gracefully.
    llvm::report_fatal_error("couldn't find enzyme kernel");
  }

  kernel->call(out, ins + 1);
}

PYBIND11_MODULE(enzyme_call, m) {
  m.def("create_enzyme_cpu_kernel",
        [](const std::string &source, const pybind11::list &out_shapes,
           bool dump_ir) -> int64_t {
          llvm::SmallVector<llvm::SmallVector<int64_t>> shapes;
          shapes.reserve(pybind11::len(out_shapes));
          for (const auto &element : out_shapes) {
            auto nested = element.cast<pybind11::list>();
            llvm::SmallVector<int64_t> &target = shapes.emplace_back();
            target.reserve(pybind11::len(nested));
            for (const auto &nested_element : nested) {
              target.push_back(nested_element.cast<int64_t>());
            }
          }
          return CpuKernel::create(source, shapes, dump_ir);
        });

  m.def("get_cpu_callback", []() {
    return pybind11::capsule(reinterpret_cast<void *>(&CpuCallback),
                             "xla._CUSTOM_CALL_TARGET");
  });
}

