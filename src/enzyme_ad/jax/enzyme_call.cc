//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/TransformOps/TransformOps.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <regex>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "clang_compile.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/RegisterEHFrames.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/RWMutex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/InitAllPasses.h"
#include "xla/mlir_hlo/deallocation/transforms/passes.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir_hlo/transforms/passes.h"

#include "xla/mlir_hlo/transforms/passes.h"

#include "compile_with_xla.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/cpu/cpu_executable.h"

#include "Enzyme/FunctionUtils.h"
#include "Enzyme/MLIR/Passes/Passes.h"

#include "nanobind/nanobind.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/tuple.h"

#include "stablehlo/transforms/Passes.h"

#include "Integrations/c/EnzymeXLA.h"

namespace {
class CpuKernel {
  // static llvm::orc::ExecutionSession ES;
  static std::unique_ptr<llvm::DataLayout> DL;
  static std::unique_ptr<llvm::orc::LLJIT> JIT;

  int64_t identifier;
  size_t num_out;
  uint64_t addr;

public:
  static constexpr size_t UNKNOWN_PLATFORM = 0x1000000000;

  CpuKernel(int64_t identifier, size_t num_out, uint64_t addr)
      : identifier(identifier), num_out(num_out), addr(addr) {}

  static std::pair<size_t, size_t>
  tapeAndTempSize(std::string fn, llvm::StringRef source,
                  llvm::ArrayRef<llvm::SmallVector<int64_t>> out_shapes,
                  llvm::ArrayRef<std::string> out_names,
                  llvm::ArrayRef<llvm::SmallVector<int64_t>> in_shapes,
                  llvm::ArrayRef<std::string> in_names, PyObject *pyargv,
                  Language lang, bool xla_runtime,
                  const std::string &pass_pipeline) {
    auto mode = ABI::Tape;
    std::vector<std::string> pyargv_strs;
    if (pyargv && PySequence_Check(pyargv)) {
      auto sz = PySequence_Size(pyargv);
      for (Py_ssize_t i = 0; i < sz; ++i) {
        PyObject *item = PySequence_GetItem(pyargv, i);
#if PY_VERSION_HEX < 0x03000000
        auto argv = PyString_AsString(item);
#else
        auto argv = PyUnicode_AsUTF8(item);
#endif
        Py_DECREF(item);
        if (argv)
          pyargv_strs.emplace_back(argv);
#if PY_VERSION_HEX < 0x03000000
        free(argv);
#endif
      }
    }
    auto mod_or_err =
        createLLVMMod(fn, source, out_shapes, out_names, in_shapes, in_names,
                      pyargv_strs, mode, lang, xla_runtime, pass_pipeline);
    if (!mod_or_err.ok()) {
      throw ::nanobind::value_error(mod_or_err.status().ToString().c_str());
    }

    auto [mod, llvm_ctx, num_out, tmpBuf] = std::move(mod_or_err).value();
    auto lfn = mod->getFunction("entry");
    auto RI =
        llvm::cast<llvm::ReturnInst>(lfn->getEntryBlock().getTerminator());
    auto val = llvm::cast<llvm::ConstantInt>(RI->getReturnValue());
    size_t res = val->getZExtValue();
    // force deletion of mod first explicitly
    mod = nullptr;
    return std::make_pair(res, tmpBuf);
  }

  static size_t tempSize(llvm::StringRef source, Language lang,
                         bool xla_runtime, const std::string &pass_pipeline) {
    switch (lang) {
    case Language::MHLO: {
      std::string llvm_ir;
      auto exec_or_err = compile_mhlo_to_llvm_with_xla(
          source, llvm_ir, xla_runtime, pass_pipeline);
      if (!exec_or_err.ok()) {
        throw ::nanobind::value_error(exec_or_err.status().ToString().c_str());
      }
      auto local_executable = std::move(exec_or_err).value();
      auto *cpu_executable = static_cast<xla::cpu::CpuExecutable *>(
          local_executable->executable());
      auto &assignment = cpu_executable->buffer_assignment();
      return assignment.temp_allocation_total_size();
    }
    default:
      return 0;
    }
  }

  static std::tuple<size_t, size_t>
  create(std::string fn, llvm::StringRef source,
         llvm::ArrayRef<llvm::SmallVector<int64_t>> out_shapes,
         llvm::ArrayRef<std::string> out_names,
         llvm::ArrayRef<llvm::SmallVector<int64_t>> in_shapes,
         llvm::ArrayRef<std::string> in_names, PyObject *pyargv, ABI mode,
         Language lang, bool xla_runtime, const std::string &pass_pipeline,
         const std::string &platform) {
    if (platform != "cpu")
      return std::make_tuple(UNKNOWN_PLATFORM, 0);
    llvm::sys::SmartScopedWriter<true> lock(kernel_mutex);
    size_t identifier = last_identifier++;

    std::vector<std::string> pyargv_strs;
    if (pyargv && PySequence_Check(pyargv)) {
      auto sz = PySequence_Size(pyargv);
      for (Py_ssize_t i = 0; i < sz; ++i) {
        PyObject *item = PySequence_GetItem(pyargv, i);
#if PY_VERSION_HEX < 0x03000000
        auto argv = PyString_AsString(item);
#else
        auto argv = PyUnicode_AsUTF8(item);
#endif
        Py_DECREF(item);
        if (argv)
          pyargv_strs.emplace_back(argv);
#if PY_VERSION_HEX < 0x03000000
        free(argv);
#endif
      }
    }
    auto mod_or_err =
        createLLVMMod(fn, source, out_shapes, out_names, in_shapes, in_names,
                      pyargv_strs, mode, lang, xla_runtime, pass_pipeline);
    if (!mod_or_err.ok()) {
      throw nanobind::value_error(mod_or_err.status().ToString().c_str());
    }
    auto [mod, llvm_ctx, num_out, tmpBuf] = std::move(mod_or_err).value();
    if (!JIT) {
      DL = std::make_unique<llvm::DataLayout>(mod->getDataLayoutStr());
      auto tJIT =
          llvm::orc::LLJITBuilder()
              .setDataLayout(*DL.get())
              .setLinkProcessSymbolsByDefault(true)
              .setObjectLinkingLayerCreator(
                  [](llvm::orc::ExecutionSession &ES)
                      -> llvm::Expected<
                          std::unique_ptr<llvm::orc::ObjectLayer>> {
                    auto obj = std::make_unique<
                        llvm::orc::RTDyldObjectLinkingLayer>(
                        ES, [](const llvm::MemoryBuffer &) {
                          return std::make_unique<llvm::SectionMemoryManager>();
                        });
                    if (getenv("ENABLE_GDBLISTENER")) {
                      auto list = llvm::JITEventListener::
                          createGDBRegistrationListener();
                      obj->registerJITEventListener(*list);
                    }
                    return obj;
                  })
              .setJITTargetMachineBuilder(llvm::orc::JITTargetMachineBuilder(
                  llvm::Triple(mod->getTargetTriple())))
              .create();
      if (!tJIT) {
        llvm::errs() << tJIT.takeError() << "\n";
        throw nanobind::value_error("failed to create jit");
      }
      JIT = std::move(tJIT.get());
      assert(JIT);
    }

    auto LibA = JIT->createJITDylib("enzymedl_" + std::to_string(identifier));

    // Add the module.
    // if (auto Err =
    // JIT->addIRModule(llvm::orc::ThreadSafeModule(std::move(mod),
    // std::move(llvm_ctx)))) {
    if (auto Err = JIT->addIRModule(
            LibA.get(),
            llvm::orc::ThreadSafeModule(std::move(mod), std::move(llvm_ctx)))) {
      llvm::errs() << " error " << Err << "\n";
      throw nanobind::value_error("failed to add IR module");
    }

    // Look up the JIT'd code entry point.
    auto EntrySym = JIT->lookup(LibA.get(), "entry");
    if (!EntrySym) {
      llvm::errs() << EntrySym.takeError() << "\n";
      throw nanobind::value_error("failed to lookup function called 'entry'");
    }

    // Cast the entry point address to a function pointer.
    auto Entry = EntrySym->getValue();

    kernels.try_emplace(
        identifier, std::make_unique<CpuKernel>(identifier, num_out, Entry));
    return std::make_tuple(identifier, tmpBuf);
  }

  static CpuKernel *get(int64_t identifier) {
    llvm::sys::SmartScopedReader<true> lock(kernel_mutex);
    auto it = kernels.find(identifier);
    if (it == kernels.end())
      return nullptr;
    return it->getSecond().get();
  }

  void call(void *out, void **ins) const {
    void **outs = num_out > 1 ? reinterpret_cast<void **>(out) : &out;
    for (int i = 0; i < num_out; i++) {
      void *data = outs[i];
    }
    auto fn = (void (*)(void **outs, void **ins))addr;
    fn(outs, ins);
  }

private:
  static llvm::DenseMap<int64_t, std::unique_ptr<CpuKernel>> kernels;
  static size_t last_identifier;
  static llvm::sys::SmartRWMutex<true> kernel_mutex;
};

llvm::DenseMap<int64_t, std::unique_ptr<CpuKernel>> CpuKernel::kernels;
size_t CpuKernel::last_identifier = 1;
llvm::sys::SmartRWMutex<true> CpuKernel::kernel_mutex;
std::unique_ptr<llvm::DataLayout> CpuKernel::DL;
std::unique_ptr<llvm::orc::LLJIT> CpuKernel::JIT = nullptr;
// llvm::orc::ExecutionSession
// CpuKernel::ES(std::move(*llvm::orc::SelfExecutorProcessControl::Create()));
} // namespace

void Callback(void *out, void **ins, void *opaque, size_t opaque_len,
              void *status) {
  int64_t identifier = *reinterpret_cast<int64_t *>(ins[0]);
  CpuKernel *kernel = CpuKernel::get(identifier);
  if (!kernel) {
    if (identifier == CpuKernel::UNKNOWN_PLATFORM) {
      throw nanobind::value_error(
          "Unknown platform callback could not be executed");
    }
    // TODO: find a way to fail more gracefully.
    llvm::report_fatal_error("couldn't find enzyme kernel");
  }
  kernel->call(out, ins + 1);
}

extern "C" void RegisterEnzymeXLAGPUHandler();
extern "C" void RegisterEnzymeXLACPUHandler();
extern "C" void registerEnzymeJaXXLAFFI();

NB_MODULE(enzyme_call, m) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();
  EnzymeAlwaysInlineDiff.setValue(true);

  mlir::registerAllPasses();

  mlir::mhlo::registerAllMhloPasses();
  // xla::cpu::registerCpuTransformsPasses();
  mlir::hlo::registerLMHLOTransformsPasses();
  // xla::runtime::registerRuntimeTransformsPasses();
  // xla::registerMemrefTransformsPasses();

  mlir::registerShapePasses();
  mlir::registerConvertShapeToStandardPass();
  mlir::registerConvertShapeConstraintsPass();
  mlir::memref::registerResolveShapedTypeResultDimsPass();
  mlir::registerLinalgPasses();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerConversionPasses();
  mlir::bufferization::registerBufferizationPasses();
  mlir::registerAsyncPasses();
  mlir::arith::registerArithPasses();
  mlir::memref::registerMemRefPasses();
  mlir::enzyme::registerenzymePasses();
  mlir::enzyme::registerenzymexlaPasses();
  mlir::enzyme::registerGenerateApplyPatternsPass();
  mlir::enzyme::registerRemoveTransformPass();
  mlir::stablehlo::registerPasses();

  nanobind::enum_<Language>(m, "Language")
      .value("CPP", Language::CPP)
      .value("LLVM", Language::LLVM)
      .value("MHLO", Language::MHLO);

  nanobind::enum_<ABI>(m, "ABI")
      .value("Primal", ABI::Primal)
      .value("Forward", ABI::Forward)
      .value("Augmented", ABI::Augmented)
      .value("Reverse", ABI::Reverse)
      .value("Tape", ABI::Tape);

  m.def("create_enzyme_kernel",
        [](const std::string &source, const std::string &fn,
           const nanobind::list &py_out_shapes,
           const nanobind::list &py_in_shapes, nanobind::object pyargv,
           ABI mode, Language lang, bool xla_runtime,
           const std::string &pass_pipeline,
           const std::string &platform) -> std::tuple<size_t, size_t> {
          llvm::SmallVector<llvm::SmallVector<int64_t>> out_shapes;
          out_shapes.reserve(nanobind::len(py_out_shapes));
          llvm::SmallVector<llvm::SmallVector<int64_t>> in_shapes;
          in_shapes.reserve(nanobind::len(py_in_shapes));

          llvm::SmallVector<std::string> out_types;
          out_types.reserve(nanobind::len(py_out_shapes));

          llvm::SmallVector<std::string> in_types;
          in_types.reserve(nanobind::len(py_in_shapes));

          for (const auto &element : py_out_shapes) {
            auto se = nanobind::cast<nanobind::tuple>(element);
            auto dtype = nanobind::cast<std::string>(se[0]);
            out_types.push_back(dtype);
            auto nested = nanobind::cast<nanobind::list>(se[1]);
            llvm::SmallVector<int64_t> &target = out_shapes.emplace_back();
            target.reserve(nanobind::len(nested));
            for (const auto &nested_element : nested) {
              target.push_back(nanobind::cast<int64_t>(nested_element));
            }
          }
          for (const auto &element : py_in_shapes) {
            auto se = nanobind::cast<nanobind::tuple>(element);
            auto dtype = nanobind::cast<std::string>(se[0]);
            in_types.push_back(dtype);
            auto nested = nanobind::cast<nanobind::list>(se[1]);
            llvm::SmallVector<int64_t> &target = in_shapes.emplace_back();
            target.reserve(nanobind::len(nested));
            for (const auto &nested_element : nested) {
              target.push_back(nanobind::cast<int64_t>(nested_element));
            }
          }
          return CpuKernel::create(fn, source, out_shapes, out_types, in_shapes,
                                   in_types, pyargv.ptr(), mode, (Language)lang,
                                   xla_runtime, pass_pipeline, platform);
        });

  m.def("tmp_size",
        [](const std::string &source, Language lang, bool xla_runtime,
           const std::string &pass_pipeline) -> size_t {
          return CpuKernel::tempSize(source, (Language)lang, xla_runtime,
                                     pass_pipeline);
        });

  m.def("compile_to_llvm",
        [](const std::string outfile, const std::string &source,
           const std::string &fn, const nanobind::list &py_out_shapes,
           const nanobind::list &py_in_shapes, nanobind::object pyargv,
           Language lang, bool xla_runtime, const std::string &pass_pipeline) {
          llvm::SmallVector<llvm::SmallVector<int64_t>> out_shapes;
          out_shapes.reserve(nanobind::len(py_out_shapes));
          llvm::SmallVector<llvm::SmallVector<int64_t>> in_shapes;
          in_shapes.reserve(nanobind::len(py_in_shapes));

          llvm::SmallVector<std::string> out_types;
          out_types.reserve(nanobind::len(py_out_shapes));

          llvm::SmallVector<std::string> in_types;
          in_types.reserve(nanobind::len(py_in_shapes));

          for (const auto &element : py_out_shapes) {
            auto se = nanobind::cast<nanobind::tuple>(element);
            auto dtype = nanobind::cast<std::string>(se[0]);
            out_types.push_back(dtype);
            auto nested = nanobind::cast<nanobind::list>(se[1]);
            llvm::SmallVector<int64_t> &target = out_shapes.emplace_back();
            target.reserve(nanobind::len(nested));
            for (const auto &nested_element : nested) {
              target.push_back(nanobind::cast<int64_t>(nested_element));
            }
          }
          for (const auto &element : py_in_shapes) {
            auto se = nanobind::cast<nanobind::tuple>(element);
            auto dtype = nanobind::cast<std::string>(se[0]);
            in_types.push_back(dtype);
            auto nested = nanobind::cast<nanobind::list>(se[1]);
            llvm::SmallVector<int64_t> &target = in_shapes.emplace_back();
            target.reserve(nanobind::len(nested));
            for (const auto &nested_element : nested) {
              target.push_back(nanobind::cast<int64_t>(nested_element));
            }
          }

          std::error_code EC;
          llvm::raw_fd_ostream ostream(outfile, EC);

          std::vector<std::string> pyargv_strs;
          auto pyargv_ptr = pyargv.ptr();
          if (pyargv_ptr && PySequence_Check(pyargv_ptr)) {
            auto sz = PySequence_Size(pyargv_ptr);
            for (Py_ssize_t i = 0; i < sz; ++i) {
              PyObject *item = PySequence_GetItem(pyargv_ptr, i);
#if PY_VERSION_HEX < 0x03000000
              auto argv = PyString_AsString(item);
#else
              auto argv = PyUnicode_AsUTF8(item);
#endif
              Py_DECREF(item);
              if (argv)
                pyargv_strs.emplace_back(argv);
#if PY_VERSION_HEX < 0x03000000
              free(argv);
#endif
            }
          }

          auto mod_or_err = createLLVMMod(
              fn, source, out_shapes, out_types, in_shapes, in_types,
              pyargv_strs, ABI::Primal, lang, xla_runtime, pass_pipeline);
          if (!mod_or_err.ok()) {
            throw nanobind::value_error(mod_or_err.status().ToString().c_str());
          }
          auto [mod, llvm_ctx, num_out, tmpBuf] = std::move(mod_or_err).value();
          ostream << *mod;
          ostream.close();
          return;
        });

  m.def("tape_and_tmp_size",
        [](const std::string &source, const std::string &fn,
           const nanobind::list &py_out_shapes,
           const nanobind::list &py_in_shapes, nanobind::object pyargv,
           Language lang, bool xla_runtime,
           const std::string &pass_pipeline) -> std::pair<size_t, size_t> {
          llvm::SmallVector<llvm::SmallVector<int64_t>> out_shapes;
          out_shapes.reserve(nanobind::len(py_out_shapes));
          llvm::SmallVector<llvm::SmallVector<int64_t>> in_shapes;
          in_shapes.reserve(nanobind::len(py_in_shapes));

          llvm::SmallVector<std::string> out_types;
          out_types.reserve(nanobind::len(py_out_shapes));

          llvm::SmallVector<std::string> in_types;
          in_types.reserve(nanobind::len(py_in_shapes));

          for (const auto &element : py_out_shapes) {
            auto se = nanobind::cast<nanobind::tuple>(element);
            auto dtype = nanobind::cast<std::string>(se[0]);
            out_types.push_back(dtype);
            auto nested = nanobind::cast<nanobind::tuple>(se[1]);
            llvm::SmallVector<int64_t> &target = out_shapes.emplace_back();
            target.reserve(nanobind::len(nested));
            for (const auto &nested_element : nested) {
              target.push_back(nanobind::cast<int64_t>(nested_element));
            }
          }
          for (const auto &element : py_in_shapes) {
            auto se = nanobind::cast<nanobind::tuple>(element);
            auto dtype = nanobind::cast<std::string>(se[0]);
            in_types.push_back(dtype);
            auto nested = nanobind::cast<nanobind::tuple>(se[1]);
            llvm::SmallVector<int64_t> &target = in_shapes.emplace_back();
            target.reserve(nanobind::len(nested));
            for (const auto &nested_element : nested) {
              target.push_back(nanobind::cast<int64_t>(nested_element));
            }
          }
          return CpuKernel::tapeAndTempSize(
              fn, source, out_shapes, out_types, in_shapes, in_types,
              pyargv.ptr(), (Language)lang, xla_runtime, pass_pipeline);
        });

  m.def("get_callback", []() {
    return nanobind::capsule(reinterpret_cast<void *>(&Callback),
                             "xla._CUSTOM_CALL_TARGET");
  });

  m.def("optimize_module",
        [](MlirModule cmod, const std::string &pass_pipeline) {
          absl::Status status = run_pass_pipeline(unwrap(cmod), pass_pipeline);
          if (!status.ok()) {
            throw nanobind::value_error(status.ToString().c_str());
          }
        });
  m.def("run_pass_pipeline", [](nanobind::object pyoldsyms,
                                const std::string &mlir,
                                const std::string &pass_pipeline) {
    auto pyargv = pyoldsyms.ptr();
    std::vector<std::string> oldsyms;
    assert(PySequence_Check(pyargv));
    auto sz = PySequence_Size(pyargv);
    for (Py_ssize_t i = 0; i < sz; ++i) {
      PyObject *item = PySequence_GetItem(pyargv, i);
#if PY_VERSION_HEX < 0x03000000
      auto argv = PyString_AsString(item);
#else
      auto argv = PyUnicode_AsUTF8(item);
#endif
      Py_DECREF(item);
      assert(argv);
      oldsyms.emplace_back(argv);
#if PY_VERSION_HEX < 0x03000000
      free(argv);
#else
      // should not free py3+
#endif
    }
    auto result_or_err = run_pass_pipeline(oldsyms, mlir, pass_pipeline);
    if (!result_or_err.ok()) {
      throw nanobind::value_error(result_or_err.status().ToString().c_str());
    }
    return result_or_err.value();
  });

  m.def("register_enzymexla_cpu_handler",
        []() { RegisterEnzymeXLACPUHandler(); });

  m.def("register_enzymexla_gpu_handler",
        []() { RegisterEnzymeXLAGPUHandler(); });

  m.def("register_enzymexla_xla_ffi", []() { registerEnzymeJaXXLAFFI(); });

  m.def("compile_mhlo_to_llvm_with_xla", [](const std::string &mhlo_text,
                                            bool xla_runtime,
                                            const std::string &pass_pipeline) {
    std::string llvm_ir;
    auto exec_or_err = compile_mhlo_to_llvm_with_xla(
        mhlo_text, llvm_ir, xla_runtime, pass_pipeline);
    if (!exec_or_err.ok()) {
      throw nanobind::value_error(exec_or_err.status().ToString().c_str());
    }
    return llvm_ir;
  });

  m.def("get_transform_passes_list",
        [](int64_t max_constant_threshold, int64_t while_unroll_threshold,
           int reshape_propagate, int transpose_propagate, bool no_nan,
           bool all_finite, bool dus_to_concat, bool dus_slice_simplify,
           bool sum_to_reducewindow, bool sum_to_conv,
           bool aggressive_sum_to_conv, bool while_concat,
           bool aggressive_propagation, bool is_sharded,
           bool raise_shlo_to_blas_lapack, bool recognize_comms,
           bool lower_comms, bool enable_structured_tensors_detection_passes,
           bool enable_structured_tensors_passes,
           bool enable_scatter_gather_optimization_passes,
           bool enable_slice_to_batch_passes,
           bool enable_reduce_slice_fusion_passes,
           bool enable_concat_to_batch_passes, bool enable_loop_raising_passes,
           bool enable_licm_optimization_passes,
           bool enable_pad_optimization_passes,
           bool enable_self_to_convolution_like_passes)
            -> std::pair<std::string, std::string> {
          EnzymeXLATransformPassesOptions options;
          options.max_constant_threshold = max_constant_threshold;
          options.while_unroll_threshold = while_unroll_threshold;
          options.reshape_propagate =
              (EnzymeXLAPropagateDirection)reshape_propagate;
          options.transpose_propagate =
              (EnzymeXLAPropagateDirection)transpose_propagate;
          options.no_nan = no_nan;
          options.all_finite = all_finite;
          options.dus_to_concat = dus_to_concat;
          options.dus_slice_simplify = dus_slice_simplify;
          options.sum_to_reducewindow = sum_to_reducewindow;
          options.sum_to_conv = sum_to_conv;
          options.aggressive_sum_to_conv = aggressive_sum_to_conv;
          options.while_concat = while_concat;
          options.aggressive_propagation = aggressive_propagation;
          options.is_sharded = is_sharded;
          options.raise_shlo_to_blas_lapack = raise_shlo_to_blas_lapack;
          options.recognize_comms = recognize_comms;
          options.lower_comms = lower_comms;
          options.enable_structured_tensors_detection_passes =
              enable_structured_tensors_detection_passes;
          options.enable_structured_tensors_passes =
              enable_structured_tensors_passes;
          options.enable_scatter_gather_optimization_passes =
              enable_scatter_gather_optimization_passes;
          options.enable_slice_to_batch_passes = enable_slice_to_batch_passes;
          options.enable_reduce_slice_fusion_passes =
              enable_reduce_slice_fusion_passes;
          options.enable_concat_to_batch_passes = enable_concat_to_batch_passes;
          options.enable_loop_raising_passes = enable_loop_raising_passes;
          options.enable_licm_optimization_passes =
              enable_licm_optimization_passes;
          options.enable_pad_optimization_passes =
              enable_pad_optimization_passes;
          options.enable_self_to_convolution_like_passes =
              enable_self_to_convolution_like_passes;

          char *mainPasses = nullptr;
          char *lowerPasses = nullptr;
          enzymexlaGetTransformPassesList(&options, &mainPasses, &lowerPasses);

          std::string main_str(mainPasses ? mainPasses : "");
          std::string lower_str(lowerPasses ? lowerPasses : "");

          enzymexlaFreeTransformPassesList(mainPasses);
          enzymexlaFreeTransformPassesList(lowerPasses);

          return std::make_pair(main_str, lower_str);
        });
}
