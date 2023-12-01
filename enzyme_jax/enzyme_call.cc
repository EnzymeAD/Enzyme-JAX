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

#include "xla/runtime/executable.h"
#include "xla/hlo/ir/hlo_instruction.h"
#define private public
#include "xla/service/cpu/cpu_executable.h"
#undef private

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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/RWMutex.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "pybind11/pybind11.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "xla/client/client_library.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/xla_computation.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "xla/translate/mhlo_to_hlo/type_to_shape.h"

// Compile an MHLO module given as a string to LLVM IR using XLA.
std::unique_ptr<xla::LocalExecutable> compile_mhlo_to_llvm_with_xla(
    llvm::StringRef mhlo_text, std::string& output) {
  // Parse MLIR.
  mlir::MLIRContext context;
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::mhlo::MhloDialect>();
  mlir::ParserConfig parser_config(&context);
  mlir::OwningOpRef<mlir::ModuleOp> parsed_module =
      mlir::parseSourceString<mlir::ModuleOp>(mhlo_text, parser_config);

  // Convert to XLA Computation.
  xla::HloProto hlo_proto;
  mlir::ConvertMlirHloToHlo(*parsed_module, &hlo_proto,
                            /*use_tuple_args=*/false, /*return_tuple=*/false);
  xla::XlaComputation xla_computation(hlo_proto.hlo_module());

  // Extract and convert the shapes fro MHLO.
  std::vector<xla::Shape> shapes;
  mlir::SymbolTable symbol_table(*parsed_module);
  auto entry_point = symbol_table.lookup<mlir::FunctionOpInterface>("main");
  shapes.reserve(entry_point.getNumArguments());
  for (mlir::Type type : entry_point.getArgumentTypes()) {
    shapes.push_back(xla::TypeToShape(type));
  }
  std::vector<const xla::Shape *> shape_pointers;
  shape_pointers.reserve(shapes.size());
  for (xla::Shape &shape : shapes) {
    shape_pointers.push_back(&shape);
  }

  // Compile with XLA, local client means targeting CPU.
  // XXX: this is using a debug feature of XLA to preserve LLVM IR. If the
  // feature ever disappears and is not recoverable with a local patch, this
  // will have to recreate the XLA pipeline. This may also be wiser in the long
  // term so we don't waste compile time running LLVM optimizations and code
  // generation only to throw away the binary.
  absl::StatusOr<xla::LocalClient *> local_client_or_error =
      xla::ClientLibrary::GetOrCreateLocalClient();
  if (!local_client_or_error.ok()) {
    throw pybind11::value_error(local_client_or_error.status().ToString());
  }
  xla::LocalClient *local_client = local_client_or_error.value();
  xla::ExecutableBuildOptions build_options;
  build_options.mutable_debug_options()->set_xla_embed_ir_in_executable(true);
  absl::StatusOr<std::vector<std::unique_ptr<xla::LocalExecutable>>>
      local_executables =
          local_client->Compile(xla_computation, shape_pointers, build_options);
  if (!local_executables.ok()) {
    throw pybind11::value_error(local_executables.status().ToString());
  }

  // Extract the LLVM IR stored in the executable.
  std::unique_ptr<xla::LocalExecutable> local_executable = std::move(local_executables.value()[0]);
  auto *cpu_executable =
      static_cast<xla::cpu::CpuExecutable *>(local_executable->executable());
  
  output = cpu_executable->ir_module_string();
  return std::move(local_executable);
}

enum class Language : int {
  CPP = 0,
  LLVM = 1,
  MHLO = 2
};

namespace {
class CpuKernel {
  // static llvm::orc::ExecutionSession ES;
  static std::unique_ptr<llvm::DataLayout> DL;
  static std::unique_ptr<llvm::orc::LLJIT > JIT;

  int64_t identifier;
  size_t num_out;
  uint64_t addr;
 public:
  CpuKernel(int64_t identifier,
            size_t num_out, uint64_t addr)
      : identifier(identifier), num_out(num_out), addr(addr) {
  }

  static std::string make_type(std::string typenam, llvm::ArrayRef<int64_t> shape, bool constv, Language lang) {
    std::string s = std::string(constv ? "const " : "") + "enzyme::tensor<" + typenam;
    for (auto v : shape) {
      s += ", " + std::to_string(v);
    }
    return s + ">";
  }

  static std::tuple<std::unique_ptr<llvm::Module>,
                    std::unique_ptr<llvm::LLVMContext>, size_t, size_t>
  createLLVMMod(llvm::StringRef fn, llvm::StringRef source,
                llvm::ArrayRef<llvm::SmallVector<int64_t>> out_shapes,
                llvm::ArrayRef<std::string> out_names,
                llvm::ArrayRef<llvm::SmallVector<int64_t>> in_shapes,
                llvm::ArrayRef<std::string> in_names, PyObject *pyargv,
                int mode, Language lang) {
    auto llvm_ctx = std::make_unique<llvm::LLVMContext>();

    std::string input;
    llvm::raw_string_ostream ss(input);
    size_t num_out;
    ss << "#include <cstdint>\n";
    ss << "#include <enzyme/tensor>\n";
    ss << "#include <enzyme/utils>\n";

    std::unique_ptr<llvm::Module> linkMod;
    std::string stringbuf;

    size_t tmpBuf = 0;

    switch (lang) {
    case Language::CPP:
      ss << source << "\n";
      break;


    case Language::MHLO:{
      auto local_executable = compile_mhlo_to_llvm_with_xla(source, stringbuf);
      auto *cpu_executable = static_cast<xla::cpu::CpuExecutable *>(local_executable->executable());
      source = stringbuf;
      auto &assignment = cpu_executable->buffer_assignment();
      tmpBuf = assignment.temp_allocation_total_size();
      llvm::errs() << assignment.ToString() << "\n";
      // explicitly fall through
    }
    case Language::LLVM:
      llvm::SMDiagnostic Err;
      linkMod = llvm::parseIR(llvm::MemoryBufferRef(source, "<input>"), Err, *llvm_ctx);
      if (!linkMod) {
        std::string err_str;
        llvm::raw_string_ostream ss(err_str);
        Err.print("llvmsource", ss, false);
        throw pybind11::value_error("failed to compile LLVM: " + ss.str());
      }
      assert(linkMod);
      if (lang == Language::MHLO) {
        for (auto &lfn : linkMod->functions()) {
          if (lfn.empty()) continue;
          if (lfn.getLinkage() == llvm::Function::InternalLinkage)
	    continue;
	  if (fn == "mhlo_main") {
            std::string err;
            llvm::raw_string_ostream ess(err);
	    ess << " Failed to compile mhlo, found multiple functions in module:\n";
	    ess << *linkMod << "\n";
	    ess << source << "\n";
	    throw std::runtime_error(ess.str());
	  }
          fn = "mhlo_main";
          lfn.setName(fn);
          lfn.addFnAttr(llvm::Attribute::AlwaysInline);
        }
      }
      ss << " extern \"C\" void " << fn << "(void* retval, void* run_options, void* params, void* buffer_table, void* status, void* prof_counters);\n\n";

      ss << " __attribute__((always_inline)) static inline void abi_wrap(";
      bool comma = false;
        for (size_t i=0, off=0; i<out_shapes.size(); i++) {
          if (comma) ss << ", ";
          ss << " " << make_type(out_names[i], out_shapes[i], false, lang) << "& __restrict__ out_" << i;
          comma = true;
        }
      if (tmpBuf != 0) {
          if (comma) ss << ", ";
          ss << " enzyme::tensor<char, " << tmpBuf << "> & __restrict__ tmpBuf";
          comma = true;
      }
      for (size_t i=0, off=0; i<in_shapes.size(); i++) {
          if (comma) ss << ", ";
        ss << " " << make_type(in_names[i], in_shapes[i], true, lang) << "& in_" << i;
        comma = true;
      }
      ss << ") {\n";
      ss << "  void* buffers[" << (out_shapes.size() + in_shapes.size() + (tmpBuf != 0)) << "] = {";
      comma = false;
        for (size_t i=0, off=0; i<out_shapes.size(); i++) {
          if (comma) ss << ", ";
          ss << " " << "(void*)&out_" << i;
          comma = true;
        }
      for (size_t i=0, off=0; i<in_shapes.size(); i++) {
          if (comma) ss << ", ";
          ss << " " << "(void*)&in_" << i;
          comma = true;
      }
      if (tmpBuf != 0) {
          if (comma) ss << ", ";
          ss << " " << "(void*)&tmpBuf";
          comma = true;
      }
      ss << "};\n";
      ss << "  " << fn << "(nullptr, nullptr, nullptr, buffers, nullptr, nullptr);\n";
      ss << "}\n";
      fn = "abi_wrap";
    }
    if (mode != 0) {
      ss << " void entry_wrap(";
      bool comma = false;
        for (size_t i=0, off=0; i<out_shapes.size(); i++) {
          if (comma) ss << ", ";
          ss << " " << make_type(out_names[i], out_shapes[i], false, lang) << "& __restrict__ out_" << i;
          comma = true;
        }
      if (tmpBuf != 0) {
          if (comma) ss << ", ";
          ss << " enzyme::tensor<char, " << tmpBuf << "> & __restrict__ tmpBuf";
          comma = true;
      }
      for (size_t i=0, off=0; i<in_shapes.size(); i++) {
          if (comma) ss << ", ";
        ss << " " << make_type(in_names[i], in_shapes[i], true, lang) << "& in_" << i;
        comma = true;
      }
      ss << ") {\n";
      ss << "  " << fn << "(";
      comma = false;
        for (size_t i=0, off=0; i<out_shapes.size(); i++) {
          if (comma) ss << ", ";
          ss << " " << "out_" << i;
          comma = true;
        }
        if (tmpBuf != 0) {
          if (comma) ss << ", ";
          ss << " " << "tmpBuf";
          comma = true;
        }
      for (size_t i=0, off=0; i<in_shapes.size(); i++) {
          if (comma) ss << ", ";
          ss << " " << "in_" << i;
          comma = true;
      }
      ss << ");\n";
      ss << "}\n";
      fn = "entry_wrap";
    }
    if (mode == 4)
      ss << "extern \"C\" std::size_t entry() {\n";
    else
      ss << "extern \"C\" void entry(void** __restrict__ outs, void** __restrict__ ins) {\n";
    size_t out_off = 0;
    size_t in_off = 0;

    if (mode == 3) {
      ss << " void*& tape = " << "*(void**)ins[" << in_off << "];\n";
      in_off++;
    }

    for (size_t i=0; i<out_shapes.size(); i++) {
      if (mode != 3 && mode != 4) {
        ss << " " << make_type(out_names[i], out_shapes[i], false, lang) << "& out_" << i << " = " << "*(" << make_type(out_names[i], out_shapes[i], false, lang) << "*)outs[" << out_off << "];\n";
        out_off++;
      }
      if (mode == 1) {
        ss << " " << make_type(out_names[i], out_shapes[i], false, lang) << "& dout_" << i << " = " << "*(" << make_type(out_names[i], out_shapes[i], false, lang) << "*)outs[" << out_off << "];\n";
        out_off++;
      }
      if (mode == 3) {
        ss << " " << make_type(out_names[i], out_shapes[i], true, lang) << "& dout_" << i << " = " << "*(" << make_type(out_names[i], out_shapes[i], true, lang) << "*)ins[" << in_off << "];\n";
        in_off++;
      }
    }

    for (size_t i=0, off=0; i<in_shapes.size(); i++) {
      if (mode != 3 && mode != 4) {
        ss << " " << make_type(in_names[i], in_shapes[i], true, lang) << "& in_" << i << " = " << "*(" << make_type(in_names[i], in_shapes[i], true, lang) << "*)ins[" << in_off << "];\n";
        in_off++;
      }
      if (mode == 1) {
        ss << " " << make_type(in_names[i], in_shapes[i], true, lang) << "& din_" << i << " = " << "*(" << make_type(in_names[i], in_shapes[i], true, lang) << "*)ins[" << in_off << "];\n";
        in_off++;
      }
      if (mode == 3) {
        ss << " " << make_type(in_names[i], in_shapes[i], false, lang) << "& din_" << i << " = " << "*(" << make_type(in_names[i], in_shapes[i], false, lang) << "*)outs[" << out_off << "];\n";
        out_off++;
      }
    }
    if (mode == 2) {
      ss << " void*& tape = " << "*(void**)outs[" << out_off << "];\n";
      out_off++;
    }
      if (mode != 4 && tmpBuf != 0) {
        ss << " enzyme::tensor<char, " << tmpBuf << ">& tmpBuf = " << "*(enzyme::tensor<char, " << tmpBuf << ">*)outs[" << out_off << "];\n";
        out_off++;
      }
      // forward mode, we have undef dtmpbuf
      if (mode == 1 && tmpBuf != 0) {
        ss << " enzyme::tensor<char, " << tmpBuf << ">& dtmpBuf = " << "*(enzyme::tensor<char, " << tmpBuf << ">*)outs[" << out_off << "];\n";
        out_off++;
      }
      // augmented forward mode, we have nullptr dtmpBuf
      if (mode == 1 && tmpBuf != 0) {
        ss << " enzyme::tensor<char, " << tmpBuf << ">& dtmpBuf = " << "*(enzyme::tensor<char, " << tmpBuf << ">*)(nullptr);\n";
      }
      // reverse mode, we have zero'd
      if (mode == 1 && tmpBuf != 0) {
        ss << " enzyme::tensor<char, " << tmpBuf << ">& dtmpBuf = " << "*(enzyme::tensor<char, " << tmpBuf << ">*)outs[" << in_off << "];\n";
        in_off++;
      }

    if (mode == 0) {
      num_out = out_shapes.size();
    ss << "  " << fn << "(";
    bool comma = false;
    for (size_t i=0; i<out_shapes.size(); i++) {
        if (comma) ss << ", ";
        ss << "out_" << i;
        comma = true;
    }
    if (tmpBuf != 0) {
        if (comma) ss << ", ";
        ss << "tmpBuf";
        comma = true;
    }
    for (size_t i=0; i<in_shapes.size(); i++) {
        if (comma) ss << ", ";
        ss << "in_" << i;
        comma = true;
    }
    ss << ");\n";
    } else if (mode == 1) {
      num_out = 2 * out_shapes.size();
      ss << "  enzyme::__enzyme_fwddiff(" << fn;
      for (size_t i=0; i<out_shapes.size(); i++) {
          ss << ", enzyme_dup, ";
          ss << "&out_" << i << ", ";
          ss << "&dout_" << i;
      }
      if (tmpBuf != 0) {
        ss << ", enzyme_dup, tmpBuf, dtmpBuf";
      }
      for (size_t i=0; i<in_shapes.size(); i++) {
          ss << ", enzyme_dup, ";
          ss << "&in_" << i << ", ";
          ss << "&din_" << i;
      }
      ss << ");\n";
    } else if (mode == 2) {
      // outs, tapeout
      // ins
      num_out = out_shapes.size() + 1 /*tape*/;
      ss << "  std::size_t tapesize = enzyme::__enzyme_augmentsize(" << fn;
      for (size_t i=0; i<out_shapes.size(); i++) {
          ss << ", enzyme_dup";
      }
      if (tmpBuf != 0) {
        ss << ", enzyme_dup,";
      }
      for (size_t i=0; i<in_shapes.size(); i++) {
          ss << ", enzyme_dup";
      }
      ss << ");\n";
      ss << "  enzyme::__enzyme_augmentfwd<void*>(" << fn << ", enzyme_allocated, tapesize, enzyme_tape, &tape";
      for (size_t i=0; i<out_shapes.size(); i++) {
          ss << ", enzyme_dup, &out_" << i << ", nullptr";
      }
      if (tmpBuf != 0) {
        ss << ", enzyme_dup, tmpBuf, dtmpBuf";
      }
      for (size_t i=0; i<in_shapes.size(); i++) {
          ss << ", enzyme_dup, &in_" << i << ", nullptr";
      }
      ss << ");\n";
    } else if (mode == 3) {
      num_out = in_shapes.size();

      // d_ins
      // tape, d_out

      // og outputs, og inputs
      //     doutputs (in), dinputs (out)
      ss << "  std::size_t tapesize = enzyme::__enzyme_augmentsize(" << fn;
      for (size_t i=0; i<out_shapes.size(); i++) {
          ss << ", enzyme_dup";
      }
      if (tmpBuf != 0) {
        ss << ", enzyme_dup";
      }
      for (size_t i=0; i<in_shapes.size(); i++) {
          ss << ", enzyme_dup";
      }
      ss << ");\n";
      for (size_t i=0; i<in_shapes.size(); i++) {
          ss << "  din_" << i << " = (" << in_names[i]<<")0;\n";
      }
      ss << "  enzyme::__enzyme_reverse<void>(" << fn << ", enzyme_allocated, tapesize, enzyme_tape, &tape";
      for (size_t i=0; i<out_shapes.size(); i++) {
          ss << ", enzyme_dup, nullptr, &dout_" << i;
      }
      if (tmpBuf != 0) {
        ss << ", enzyme_dup, tmpBuf, dtmpBuf";
      }
      for (size_t i=0; i<in_shapes.size(); i++) {
          ss << ", enzyme_dup, nullptr, &din_" << i;
      }
      ss << ");\n";
    } else if (mode == 4) {
      // outs, tapeout
      // ins
      num_out = out_shapes.size() + 1 /*tape*/;
      ss << "  std::size_t tapesize = enzyme::__enzyme_augmentsize(" << fn;
      for (size_t i=0; i<out_shapes.size(); i++) {
          ss << ", enzyme_dup";
      }
      if (tmpBuf != 0) {
        ss << ", enzyme_dup";
      }
      for (size_t i=0; i<in_shapes.size(); i++) {
          ss << ", enzyme_dup";
      }
      ss << ");\n";
      ss << "  return tapesize;\n";
    } else {
      assert(0 && "unhandled mode");
    }
    ss << "}\n";

    llvm::SmallVector<std::string> pyargv_strs;
  assert (PySequence_Check(pyargv));
	auto sz = PySequence_Size(pyargv);
    for (Py_ssize_t i = 0; i < sz; ++i) {
        PyObject* item = PySequence_GetItem(pyargv, i);
#if PY_VERSION_HEX < 0x03000000
        auto argv = PyString_AsString(item);
#else
        auto argv = PyUnicode_AsUTF8(item);
#endif
        Py_DECREF(item);
		assert(argv);
		pyargv_strs.emplace_back(argv);
#if PY_VERSION_HEX < 0x03000000
        free(argv);
#else
        // should not free py3+
#endif
    }

    auto mod = GetLLVMFromJob("/enzyme_call/source.cpp", ss.str(), /*cpp*/true, pyargv_strs, llvm_ctx.get(), std::move(linkMod));
    if (!mod)
      throw pybind11::value_error("failed to compile C++");
    return std::make_tuple(std::move(mod), std::move(llvm_ctx), num_out, tmpBuf);
  }

  static std::pair<size_t, size_t> tapeAndTempSize(llvm::StringRef fn, llvm::StringRef source,
                        llvm::ArrayRef<llvm::SmallVector<int64_t>> out_shapes,
                        llvm::ArrayRef<std::string> out_names,
                        llvm::ArrayRef<llvm::SmallVector<int64_t>> in_shapes,
                        llvm::ArrayRef<std::string> in_names,
                        PyObject* pyargv, Language lang) {
    int mode = 4;
    auto [mod, llvm_ctx, num_out, tmpBuf] = createLLVMMod(fn, source, out_shapes, out_names, in_shapes, in_names, pyargv, mode, lang);
    auto lfn = mod->getFunction("entry");
    auto RI = llvm::cast<llvm::ReturnInst>(lfn->getEntryBlock().getTerminator());
    auto val = llvm::cast<llvm::ConstantInt>(RI->getReturnValue());
    size_t res = val->getZExtValue();
    // force deletion of mod first explicitly
    mod = nullptr;
    return std::make_pair(res, tmpBuf);
  }
  
  static size_t tempSize(llvm::StringRef source, Language lang) {
    switch (lang) {
    case Language::MHLO:{
      std::string llvm_ir;
      auto local_executable = compile_mhlo_to_llvm_with_xla(source, llvm_ir);
      auto *cpu_executable = static_cast<xla::cpu::CpuExecutable *>(local_executable->executable());
      auto &assignment = cpu_executable->buffer_assignment();
      return assignment.temp_allocation_total_size();
    }
    default:
      return 0;
    }
  }

  static std::tuple<size_t, size_t> create(llvm::StringRef fn, llvm::StringRef source,
                        llvm::ArrayRef<llvm::SmallVector<int64_t>> out_shapes,
                        llvm::ArrayRef<std::string> out_names,
                        llvm::ArrayRef<llvm::SmallVector<int64_t>> in_shapes,
                        llvm::ArrayRef<std::string> in_names,
                        PyObject* pyargv, int mode, Language lang) {
    llvm::sys::SmartScopedWriter<true> lock(kernel_mutex);
    size_t identifier = last_identifier++;

    auto [mod, llvm_ctx, num_out, tmpBuf] = createLLVMMod(fn, source, out_shapes, out_names, in_shapes, in_names, pyargv, mode, lang);

    if (!JIT) {
      DL = std::make_unique<llvm::DataLayout>(mod.get());
      auto tJIT = llvm::orc::LLJITBuilder().setDataLayout(*DL.get()).setLinkProcessSymbolsByDefault(true).setObjectLinkingLayerCreator(
          [](llvm::orc::ExecutionSession & ES, const llvm::Triple &OLL) -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
            return std::make_unique<llvm::orc::ObjectLinkingLayer>(ES);
          }).setJITTargetMachineBuilder(llvm::orc::JITTargetMachineBuilder(llvm::Triple(mod->getTargetTriple()))).create();
      if (!tJIT) {
        llvm::errs() << tJIT.takeError() << "\n";
        throw pybind11::value_error("failed to create jit");
      }
      JIT = std::move(tJIT.get());
      assert(JIT);
    }

    auto LibA = JIT->createJITDylib("enzymedl_"+std::to_string(identifier));

    // Add the module.
    // if (auto Err = JIT->addIRModule(llvm::orc::ThreadSafeModule(std::move(mod), std::move(llvm_ctx)))) {
    if (auto Err = JIT->addIRModule(LibA.get(), llvm::orc::ThreadSafeModule(std::move(mod), std::move(llvm_ctx)))) {
      llvm::errs() <<" error "  << Err << "\n";
      throw pybind11::value_error("failed to add IR module");
    }

    // Look up the JIT'd code entry point.
    auto EntrySym = JIT->lookup(LibA.get(), "entry");
    if (!EntrySym) {
      llvm::errs() << EntrySym.takeError() << "\n";
      throw pybind11::value_error("failed to lookup function called 'entry'");
    }

    // Cast the entry point address to a function pointer.
    auto Entry = EntrySym->getValue();
 
    kernels.try_emplace(
        identifier,
        std::make_unique<CpuKernel>(identifier, num_out, Entry));
    return std::make_tuple(identifier, tmpBuf);
  }

  static CpuKernel *get(int64_t identifier) {
    llvm::sys::SmartScopedReader<true> lock(kernel_mutex);
    auto it = kernels.find(identifier);
    if (it == kernels.end()) return nullptr;
    return it->getSecond().get();
  }

  void call(void *out, void **ins) const {
    void **outs = num_out > 1 ? reinterpret_cast<void **>(out) : &out;
    for(int i=0; i<num_out; i++) {
      void* data = outs[i];
      *(void**)(data) = 0;
    }
    auto fn = (void(*)(void**outs, void**ins))addr;
    fn(outs, ins);
  }

 private:
  static llvm::DenseMap<int64_t, std::unique_ptr<CpuKernel>> kernels;
  static size_t last_identifier;
  static llvm::sys::SmartRWMutex<true> kernel_mutex;
};

llvm::DenseMap<int64_t, std::unique_ptr<CpuKernel>>
    CpuKernel::kernels;
size_t CpuKernel::last_identifier = 1;
llvm::sys::SmartRWMutex<true> CpuKernel::kernel_mutex;
std::unique_ptr<llvm::DataLayout> CpuKernel::DL;
std::unique_ptr<llvm::orc::LLJIT > CpuKernel::JIT = nullptr;
// llvm::orc::ExecutionSession CpuKernel::ES(std::move(*llvm::orc::SelfExecutorProcessControl::Create()));
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
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  pybind11::enum_<Language>(m, "Language")
    .value("CPP", Language::CPP)
    .value("LLVM", Language::LLVM)
    .value("MHLO", Language::MHLO);

  m.def("create_enzyme_cpu_kernel",
        [](const std::string &source, const std::string &fn, const pybind11::list &py_out_shapes,
          const pybind11::list &py_in_shapes,
           pybind11::object pyargv, int mode, Language lang) -> std::tuple<size_t, size_t> {
          llvm::SmallVector<llvm::SmallVector<int64_t>> out_shapes;
          out_shapes.reserve(pybind11::len(py_out_shapes));
          llvm::SmallVector<llvm::SmallVector<int64_t>> in_shapes;
          in_shapes.reserve(pybind11::len(py_in_shapes));

          llvm::SmallVector<std::string> out_types;
          out_types.reserve(pybind11::len(py_out_shapes));

          llvm::SmallVector<std::string> in_types;
          in_types.reserve(pybind11::len(py_in_shapes));

          for (const auto &element : py_out_shapes) {
            auto se = element.cast<pybind11::tuple>();
            auto dtype = se[0].cast<std::string>();
            out_types.push_back(dtype);
            auto nested = se[1].cast<pybind11::list>();
            llvm::SmallVector<int64_t> &target = out_shapes.emplace_back();
            target.reserve(pybind11::len(nested));
            for (const auto &nested_element : nested) {
              target.push_back(nested_element.cast<int64_t>());
            }
          }
          for (const auto &element : py_in_shapes) {
            auto se = element.cast<pybind11::tuple>();
            auto dtype = se[0].cast<std::string>();
            in_types.push_back(dtype);
            auto nested = se[1].cast<pybind11::list>();
            llvm::SmallVector<int64_t> &target = in_shapes.emplace_back();
            target.reserve(pybind11::len(nested));
            for (const auto &nested_element : nested) {
              target.push_back(nested_element.cast<int64_t>());
            }
          }
          return CpuKernel::create(fn, source, out_shapes, out_types, in_shapes, in_types, pyargv.ptr(), mode, (Language)lang);
        });
  
  m.def("tmp_size",
        [](const std::string &source, Language lang) -> size_t {
          return CpuKernel::tempSize(source, (Language)lang);
        });

  m.def("tape_and_tmp_size",
        [](const std::string &source, const std::string &fn, const pybind11::list &py_out_shapes,
          const pybind11::list &py_in_shapes,
           pybind11::object pyargv, Language lang) -> std::pair<size_t, size_t> {
          llvm::SmallVector<llvm::SmallVector<int64_t>> out_shapes;
          out_shapes.reserve(pybind11::len(py_out_shapes));
          llvm::SmallVector<llvm::SmallVector<int64_t>> in_shapes;
          in_shapes.reserve(pybind11::len(py_in_shapes));

          llvm::SmallVector<std::string> out_types;
          out_types.reserve(pybind11::len(py_out_shapes));

          llvm::SmallVector<std::string> in_types;
          in_types.reserve(pybind11::len(py_in_shapes));

          for (const auto &element : py_out_shapes) {
            auto se = element.cast<pybind11::tuple>();
            auto dtype = se[0].cast<std::string>();
            out_types.push_back(dtype);
            auto nested = se[1].cast<pybind11::list>();
            llvm::SmallVector<int64_t> &target = out_shapes.emplace_back();
            target.reserve(pybind11::len(nested));
            for (const auto &nested_element : nested) {
              target.push_back(nested_element.cast<int64_t>());
            }
          }
          for (const auto &element : py_in_shapes) {
            auto se = element.cast<pybind11::tuple>();
            auto dtype = se[0].cast<std::string>();
            in_types.push_back(dtype);
            auto nested = se[1].cast<pybind11::list>();
            llvm::SmallVector<int64_t> &target = in_shapes.emplace_back();
            target.reserve(pybind11::len(nested));
            for (const auto &nested_element : nested) {
              target.push_back(nested_element.cast<int64_t>());
            }
          }
          return CpuKernel::tapeAndTempSize(fn, source, out_shapes, out_types, in_shapes, in_types, pyargv.ptr(), (Language)lang);
        });

  m.def("get_cpu_callback", []() {
    return pybind11::capsule(reinterpret_cast<void *>(&CpuCallback),
                             "xla._CUSTOM_CALL_TARGET");
  });

  m.def("compile_mhlo_to_llvm_with_xla", [](const std::string &mhlo_text) {
    std::string llvm_ir;
    compile_mhlo_to_llvm_with_xla(mhlo_text, llvm_ir);
    return llvm_ir;
  });
}

