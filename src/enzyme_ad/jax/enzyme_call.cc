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

#include "absl/status/statusor.h"
#include "clang_compile.h"
#include "pybind11/pybind11.h"
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
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
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

#include "stablehlo/transforms/Passes.h"

enum class ABI { Primal, Forward, Augmented, Reverse, Tape };

enum class Language : int { CPP = 0, LLVM = 1, MHLO = 2 };

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

  static std::string make_type(std::string typenam,
                               llvm::ArrayRef<int64_t> shape, bool constv,
                               Language lang) {
    std::string s =
        std::string(constv ? "const " : "") + "enzyme::tensor<" + typenam;
    for (auto v : shape) {
      s += ", " + std::to_string(v);
    }
    return s + ">";
  }

  static std::tuple<std::unique_ptr<llvm::Module>,
                    std::unique_ptr<llvm::LLVMContext>, size_t, size_t>
  createLLVMMod(std::string fn, llvm::StringRef source,
                llvm::ArrayRef<llvm::SmallVector<int64_t>> out_shapes,
                llvm::ArrayRef<std::string> out_names,
                llvm::ArrayRef<llvm::SmallVector<int64_t>> in_shapes,
                llvm::ArrayRef<std::string> in_names, PyObject *pyargv,
                ABI mode, Language lang, bool xla_runtime,
                const std::string &pass_pipeline) {
    auto llvm_ctx = std::make_unique<llvm::LLVMContext>();

    std::string input;
    llvm::raw_string_ostream ss(input);
    ss << "#include <cstdint>\n";
    ss << "#include <enzyme/tensor>\n";
    ss << "#include <enzyme/utils>\n";

    std::unique_ptr<llvm::Module> linkMod;
    std::unique_ptr<xla::LocalExecutable> local_executable;
    std::string stringbuf;

    size_t tmpBuf = 0;
    llvm::StringRef origSource = source;
    switch (lang) {
    case Language::CPP:
      ss << source << "\n";
      break;

    case Language::MHLO: {
      local_executable = compile_mhlo_to_llvm_with_xla(
          source, stringbuf, xla_runtime, pass_pipeline);
      auto *cpu_executable = static_cast<xla::cpu::CpuExecutable *>(
          local_executable->executable());
      auto &assignment = cpu_executable->buffer_assignment();
      if (!xla_runtime) {
        size_t num_in = 0;
        for (auto &buf2 : assignment.Allocations()) {
          if (buf2.is_entry_computation_parameter()) {
            num_in++;
          }
        }
        if (num_in != in_shapes.size()) {
          std::string err_str;
          llvm::raw_string_ostream ss(err_str);
          ss << assignment.ToString() << "\n";
          ss << source << "\n";
          ss << " Number of mhlo inputs (" << num_in
             << ") != number of jax inputs (" << in_shapes.size() << "):\n";
          ss << source << "\n";
          throw pybind11::value_error(ss.str());
        }
        for (size_t i = 0; i < in_shapes.size(); i++) {
          ssize_t idx = -1;
          for (auto &buf2 : assignment.Allocations()) {
            if (!buf2.is_entry_computation_parameter())
              continue;
            if (buf2.parameter_number() != i)
              continue;
            assert(idx == -1);
            idx = buf2.index();
          }
          if (idx == -1) {
            std::string err_str;
            llvm::raw_string_ostream ss(err_str);
            ss << " Could not find input parameter (" << i
               << ") as hlo parameter:\n";
            ss << source << "\n";
            throw pybind11::value_error(ss.str());
          }
        }
      }
      source = stringbuf;
      if (xla_runtime)
        tmpBuf = 0;
      else
        tmpBuf = assignment.temp_allocation_total_size();
      // explicitly fall through
    }
    case Language::LLVM:
      llvm::SMDiagnostic Err;
      linkMod = llvm::parseIR(llvm::MemoryBufferRef(source, "<input>"), Err,
                              *llvm_ctx);
      if (!linkMod) {
        std::string err_str;
        llvm::raw_string_ostream ss(err_str);
        Err.print("llvmsource", ss, false);
        throw pybind11::value_error("failed to compile LLVM: " + ss.str());
      }
      assert(linkMod);
      if (lang == Language::MHLO) {
        auto *cpu_executable = static_cast<xla::cpu::CpuExecutable *>(
            local_executable->executable());
        llvm::StringRef fname = cpu_executable->module_name();
        if (fname.size() && fname[0] == '_')
          fname = fname.substr(1);
        auto F = linkMod->getFunction(fname);
        if (!F) {
          llvm::errs() << *linkMod << "\n";
          llvm::errs() << "fname: " << fname << "\n";
        }
        assert(F);
        fn = "mhlo_main";
        F->setName(fn);
        assert(!F->empty());
        for (auto &F2 : *linkMod)
          if (!F2.empty()) {
            F2.addFnAttr(llvm::Attribute::AlwaysInline);
            // Remove invariant_load if we expect enzyme to cache explicitly all
            // data. Otherwise invariant_load allows Enzyme to assume it need
            // not cache, and it is illegal for us to pass in nullptr as the
            // primal (since it may be needed).
            if (mode == ABI::Augmented || mode == ABI::Reverse ||
                mode == ABI::Tape) {
              for (auto &BB : F2)
                for (auto &I : BB)
                  if (auto LI = llvm::dyn_cast<llvm::LoadInst>(&I))
                    if (LI->hasMetadata(llvm::LLVMContext::MD_invariant_load))
                      LI->setMetadata(llvm::LLVMContext::MD_invariant_load,
                                      nullptr);
            }
          }
      }
      if (xla_runtime) {
        ss << " extern \"C\" void " << fn << "(void* exec";

        for (size_t i = 0; i < in_shapes.size(); i++) {
          ss << ", void*";
          ss << ", void*";
          ss << ", int64_t";
          for (auto idx : in_shapes[i])
            ss << ", int64_t";
          for (auto idx : in_shapes[i])
            ss << ", int64_t";
        }

        for (size_t i = 0; i < out_shapes.size(); i++) {
          ss << ", void*";
          ss << ", void*";
          ss << ", int64_t";
          for (auto idx : out_shapes[i])
            ss << ", int64_t";
          for (auto idx : out_shapes[i])
            ss << ", int64_t";
        }
        ss << ");\n\n";
      } else {
        ss << " extern \"C\" void " << fn
           << "(void* retval, void* run_options, void* params, void* "
              "buffer_table, void* status, void* prof_counters);\n\n";
      }

      if (local_executable && !xla_runtime) {
        auto *cpu_executable = static_cast<xla::cpu::CpuExecutable *>(
            local_executable->executable());
        auto &assignment = cpu_executable->buffer_assignment();
        for (auto &buf : assignment.Allocations()) {
          if (!buf.is_constant())
            continue;
          assert(buf.assigned_buffers().size() == 1);
          auto hlo = buf.assigned_buffers().begin()->first;
          auto tyenum = hlo->shape().element_type();
          std::string ty;
          switch (tyenum) {
          case xla::PrimitiveType::S8:
            ty = "int8_t";
            break;
          case xla::PrimitiveType::S16:
            ty = "int16_t";
            break;
          case xla::PrimitiveType::S32:
            ty = "int32_t";
            break;
          case xla::PrimitiveType::S64:
            ty = "int64_t";
            break;
          case xla::PrimitiveType::U8:
            ty = "uint8_t";
            break;
          case xla::PrimitiveType::U16:
            ty = "uint16_t";
            break;
          case xla::PrimitiveType::U32:
            ty = "uint32_t";
            break;
          case xla::PrimitiveType::U64:
            ty = "uint64_t";
            break;
          case xla::PrimitiveType::F16:
            ty = "half";
            break;
          case xla::PrimitiveType::F32:
            ty = "float";
            break;
          case xla::PrimitiveType::F64:
            ty = "double";
            break;
          case xla::PrimitiveType::PRED:
            ty = "bool";
            break;
          default: {
            std::string err;
            llvm::raw_string_ostream ess(err);
            ess << " Failed to compile mhlo, unknown constant element type: "
                << hlo->shape().ToString() << "\n";
            throw std::runtime_error(ess.str());
          }
          }
          auto val = xla::Cast<xla::HloConstantInstruction>(hlo->instruction());

          llvm::ArrayRef<int64_t> shape(hlo->shape().dimensions().begin(),
                                        hlo->shape().dimensions().end());
          ss << "  static constexpr "
             << make_type(ty, shape, /*const*/ false, lang) << " const_"
             << buf.index() << " = ";

          xla::StringPrinter printer;
          val->literal().PrintWithoutShape(&printer);
          auto str = std::move(printer).ToString();
          if (shape.size() == 0)
            ss << "{";
          str = std::regex_replace(str, std::regex("\\{"), "{{");
          str = std::regex_replace(str, std::regex("\\}"), "}}");
          ss << str;
          if (shape.size() == 0)
            ss << "}";
          ss << ";\n";
        }
      }

      llvm::StringRef abiName = "abi_wrap";
      if (mode == ABI::Augmented)
        abiName = "aug_abi_wrap";
      else if (mode == ABI::Reverse)
        abiName = "rev_abi_wrap";
      ss << " __attribute__((always_inline)) static inline void " << abiName
         << "(";
      bool comma = false;
      for (size_t i = 0; i < out_shapes.size(); i++) {
        if (comma)
          ss << ", ";
        ss << " " << make_type(out_names[i], out_shapes[i], false, lang)
           << "& __restrict__ out_" << i;
        comma = true;
      }
      if (tmpBuf != 0) {
        if (comma)
          ss << ", ";
        ss << " enzyme::tensor<char, " << tmpBuf << "> & __restrict__ tmpBuf";
        comma = true;
      }
      for (size_t i = 0; i < in_shapes.size(); i++) {
        if (comma)
          ss << ", ";
        ss << " " << make_type(in_names[i], in_shapes[i], true, lang) << "& in_"
           << i;
        comma = true;
      }
      ss << ") {\n";
      if (xla_runtime) {
        ss << "  " << fn << "(/*exec*/nullptr";
        for (size_t i = 0; i < in_shapes.size(); i++) {
          ss << ", (void*)&in_" << i;
          ss << ", (void*)&in_" << i;
          ss << ", 0";
          for (auto idx : in_shapes[i])
            ss << ", " << idx;
          for (auto idx : in_shapes[i])
            ss << ", 0";
        }

        for (size_t i = 0; i < out_shapes.size(); i++) {
          ss << ", (void*)&out_" << i;
          ss << ", (void*)&out_" << i;
          ss << ", 0";
          for (auto idx : out_shapes[i])
            ss << ", " << idx;
          for (auto idx : out_shapes[i])
            ss << ", 0";
        }
        ss << ");\n";
      } else {
        size_t numBuffers = out_shapes.size() + in_shapes.size();
        std::vector<int> out_idxs;
        if (local_executable) {
          auto *cpu_executable = static_cast<xla::cpu::CpuExecutable *>(
              local_executable->executable());
          auto &assignment = cpu_executable->buffer_assignment();
          numBuffers = assignment.Allocations().size();
          if (out_shapes.size() == 1) {
            ssize_t idx = -1;
            for (auto &buf2 : assignment.Allocations()) {
              if (!buf2.maybe_live_out())
                continue;
              assert(!buf2.is_tuple());
              assert(idx == -1);
              idx = buf2.index();
            }
            assert(idx != -1);
            out_idxs.push_back(idx);
          } else {
            // If a tuple, find the tuple buf, then use that to index the
            // outputs.
            ssize_t tupidx = -1;
            for (auto &buf2 : assignment.Allocations()) {
              if (!buf2.maybe_live_out())
                continue;
              if (!buf2.is_tuple())
                continue;
              assert(tupidx == -1);
              tupidx = buf2.index();
            }
            assert(tupidx != -1);
            auto &tup_buf = assignment.Allocations()[tupidx];
            assert(tup_buf.assigned_buffers().size() == 1);
            auto hlo = tup_buf.assigned_buffers().begin()->first;
            auto val = hlo->instruction();
            assert(val->operand_count() == out_shapes.size());
            for (size_t i = 0; i < out_shapes.size(); i++) {
              ssize_t found = -1;
              auto operand = val->operand(i);
              while (found == -1) {
                for (auto &buf : assignment.Allocations()) {
                  if (!buf.maybe_live_out())
                    continue;
                  if (buf.is_tuple())
                    continue;
                  bool contains_output = false;
                  for (auto &pair : buf.assigned_buffers()) {
                    if (pair.first->instruction() != operand)
                      continue;
                    assert(!contains_output);
                    contains_output = true;
                    assert(pair.second.offset == 0);
                  }
                  if (!contains_output)
                    continue;
                  assert(found == -1);
                  found = buf.index();
                }
                if (operand->opcode() == xla::HloOpcode::kBitcast) {
                  operand = operand->operand(0);
                  continue;
                }
                break;
              }
              if (found == -1) {
                llvm::errs() << "assignment: " << assignment.ToString() << "\n";
                llvm::errs() << "val: " << val->ToString() << "\n";
                llvm::errs() << "vop: " << val->operand(i)->ToString() << "\n";
                llvm::errs() << "i: " << i << "\n";
              }
              assert(found != -1);
              out_idxs.push_back((int)found);
            }
          }
          for (auto &buf : assignment.Allocations()) {
            if (buf.is_thread_local()) {
              ss << "  char local_" << buf.index() << "[" << buf.size()
                 << "];\n";
              continue;
            }
            if (!buf.maybe_live_out())
              continue;
            if (!buf.is_tuple())
              continue;
            ss << "  void* tup_" << buf.index() << "[" << out_idxs.size()
               << "] = {";

            for (size_t i = 0; i < out_idxs.size(); i++) {
              if (i != 0)
                ss << ", ";
              ss << " "
                 << "(void*)&out_" << i;
            }
            ss << "};\n";
          }
        }
        ss << "  void* buffers[" << numBuffers << "] = {";

        if (local_executable) {
          auto *cpu_executable = static_cast<xla::cpu::CpuExecutable *>(
              local_executable->executable());
          auto &assignment = cpu_executable->buffer_assignment();
          for (auto &buf : assignment.Allocations()) {
            if (buf.index() != 0)
              ss << ", ";
            if (buf.is_entry_computation_parameter()) {
              ss << " "
                 << "(void*)&in_" << buf.parameter_number();
            } else if (buf.IsPreallocatedTempBuffer()) {
              ss << " "
                 << "(void*)&tmpBuf";
            } else if (buf.maybe_live_out()) {
              if (buf.is_tuple()) {
                assert(out_shapes.size() != 1);
                ss << " "
                   << "(void*)&tup_" << buf.index();
                continue;
              }
              auto it =
                  std::find(out_idxs.begin(), out_idxs.end(), buf.index());
              assert(it != out_idxs.end());
              int index = it - out_idxs.begin();
              ss << " "
                 << "(void*)&out_" << index;
            } else if (buf.is_constant()) {
              ss << " "
                 << "(void*)&const_" << buf.index();
            } else if (buf.is_thread_local()) {
              ss << " "
                 << "(void*)&local_" << buf.index();
            } else {
              std::string err;
              llvm::raw_string_ostream ess(err);
              ess << " Failed to compile mhlo, unknown buffer type\n";
              ess << origSource << "\n";
              ess << source << "\n";
              ess << local_executable->executable()->module().ToString()
                  << "\n";
              ess << " unknown buffer type: " << buf.ToString() << "\n";
              throw std::runtime_error(ess.str());
            }
          }
        } else {
          comma = false;
          for (size_t i = 0; i < out_shapes.size(); i++) {
            if (comma)
              ss << ", ";
            ss << " "
               << "(void*)&out_" << i;
            comma = true;
          }
          for (size_t i = 0; i < in_shapes.size(); i++) {
            if (comma)
              ss << ", ";
            ss << " "
               << "(void*)&in_" << i;
            comma = true;
          }
          if (tmpBuf != 0) {
            if (comma)
              ss << ", ";
            ss << " "
               << "(void*)&tmpBuf";
            comma = true;
          }
        }
        ss << "};\n";
        ss << "  " << fn
           << "(nullptr, nullptr, nullptr, buffers, nullptr, nullptr);\n";
      }
      ss << "};\n";
      fn = abiName;
    }
    if (mode != ABI::Primal) {
      ss << " void entry_wrap(";
      bool comma = false;
      for (size_t i = 0; i < out_shapes.size(); i++) {
        if (comma)
          ss << ", ";
        ss << " " << make_type(out_names[i], out_shapes[i], false, lang)
           << "& __restrict__ out_" << i;
        comma = true;
      }
      if (tmpBuf != 0) {
        if (comma)
          ss << ", ";
        ss << " enzyme::tensor<char, " << tmpBuf << "> & __restrict__ tmpBuf";
        comma = true;
      }
      for (size_t i = 0; i < in_shapes.size(); i++) {
        if (comma)
          ss << ", ";
        ss << " " << make_type(in_names[i], in_shapes[i], true, lang) << "& in_"
           << i;
        comma = true;
      }
      ss << ") {\n";
      ss << "  " << fn << "(";
      comma = false;
      for (size_t i = 0; i < out_shapes.size(); i++) {
        if (comma)
          ss << ", ";
        ss << " "
           << "out_" << i;
        comma = true;
      }
      if (tmpBuf != 0) {
        if (comma)
          ss << ", ";
        ss << " "
           << "tmpBuf";
        comma = true;
      }
      for (size_t i = 0; i < in_shapes.size(); i++) {
        if (comma)
          ss << ", ";
        ss << " "
           << "in_" << i;
        comma = true;
      }
      ss << ");\n";
      ss << "}\n";
      fn = "entry_wrap";
    }
    if (mode == ABI::Tape)
      ss << "extern \"C\" std::size_t entry() {\n";
    else
      ss << "extern \"C\" void entry(void** __restrict__ outs, void** "
            "__restrict__ ins) {\n";
    size_t out_off = 0;
    size_t in_off = 0;

    if (mode == ABI::Reverse) {
      ss << " void*& tape = "
         << "*(void**)ins[" << in_off << "];\n";
      in_off++;
    }

    for (size_t i = 0; i < out_shapes.size(); i++) {
      if (mode != ABI::Reverse && mode != ABI::Tape) {
        ss << " " << make_type(out_names[i], out_shapes[i], false, lang)
           << "& out_" << i << " = "
           << "*(" << make_type(out_names[i], out_shapes[i], false, lang)
           << "*)outs[" << out_off << "];\n";
        out_off++;
      }
      if (mode == ABI::Forward) {
        ss << " " << make_type(out_names[i], out_shapes[i], false, lang)
           << "& dout_" << i << " = "
           << "*(" << make_type(out_names[i], out_shapes[i], false, lang)
           << "*)outs[" << out_off << "];\n";
        out_off++;
      }
      if (mode == ABI::Reverse) {
        ss << " " << make_type(out_names[i], out_shapes[i], true, lang)
           << "& dout_" << i << " = "
           << "*(" << make_type(out_names[i], out_shapes[i], true, lang)
           << "*)ins[" << in_off << "];\n";
        in_off++;
      }
    }

    for (size_t i = 0; i < in_shapes.size(); i++) {
      if (mode != ABI::Reverse && mode != ABI::Tape) {
        ss << " " << make_type(in_names[i], in_shapes[i], true, lang) << "& in_"
           << i << " = "
           << "*(" << make_type(in_names[i], in_shapes[i], true, lang)
           << "*)ins[" << in_off << "];\n";
        in_off++;
      }
      if (mode == ABI::Forward) {
        ss << " " << make_type(in_names[i], in_shapes[i], true, lang)
           << "& din_" << i << " = "
           << "*(" << make_type(in_names[i], in_shapes[i], true, lang)
           << "*)ins[" << in_off << "];\n";
        in_off++;
      }
      if (mode == ABI::Reverse) {
        ss << " " << make_type(in_names[i], in_shapes[i], false, lang)
           << "& din_" << i << " = "
           << "*(" << make_type(in_names[i], in_shapes[i], false, lang)
           << "*)outs[" << out_off << "];\n";
        out_off++;
      }
    }
    if (mode == ABI::Augmented) {
      ss << " void*& tape = "
         << "*(void**)outs[" << out_off << "];\n";
      out_off++;
    }
    if (mode != ABI::Tape && mode != ABI::Reverse && tmpBuf != 0) {
      ss << " enzyme::tensor<char, " << tmpBuf << ">& tmpBuf = "
         << "*(enzyme::tensor<char, " << tmpBuf << ">*)outs[" << out_off
         << "];\n";
      out_off++;
    }
    // forward mode, we have undef dtmpbuf
    if (mode == ABI::Forward && tmpBuf != 0) {
      ss << " enzyme::tensor<char, " << tmpBuf << ">& dtmpBuf = "
         << "*(enzyme::tensor<char, " << tmpBuf << ">*)outs[" << out_off
         << "];\n";
      out_off++;
    }
    // augmented forward mode, we have nullptr dtmpBuf
    if (mode == ABI::Augmented && tmpBuf != 0) {
      ss << "#pragma clang diagnostic push\n";
      ss << "#pragma clang diagnostic ignored \"-Wnull-dereference\"\n";
      ss << " enzyme::tensor<char, " << tmpBuf << ">& dtmpBuf = "
         << "*(enzyme::tensor<char, " << tmpBuf << ">*)(nullptr);\n";
      ss << "#pragma clang diagnostic pop\n";
    }
    // reverse mode, we have zero'd
    if (mode == ABI::Reverse && tmpBuf != 0) {
      ss << "#pragma clang diagnostic push\n";
      ss << "#pragma clang diagnostic ignored \"-Wnull-dereference\"\n";
      ss << " enzyme::tensor<char, " << tmpBuf << ">& tmpBuf = "
         << "*(enzyme::tensor<char, " << tmpBuf << ">*)(nullptr);\n";
      ss << "#pragma clang diagnostic pop\n";
      ss << " __builtin_memset(outs[" << out_off << "], 0, " << tmpBuf
         << ");\n";
      ss << " enzyme::tensor<char, " << tmpBuf << ">& dtmpBuf = "
         << "*(enzyme::tensor<char, " << tmpBuf << ">*)outs[" << out_off
         << "];\n";
      out_off++;
    }

    if (mode == ABI::Primal) {
      ss << "  " << fn << "(";
      bool comma = false;
      for (size_t i = 0; i < out_shapes.size(); i++) {
        if (comma)
          ss << ", ";
        ss << "out_" << i;
        comma = true;
      }
      if (tmpBuf != 0) {
        if (comma)
          ss << ", ";
        ss << "tmpBuf";
        comma = true;
      }
      for (size_t i = 0; i < in_shapes.size(); i++) {
        if (comma)
          ss << ", ";
        ss << "in_" << i;
        comma = true;
      }
      ss << ");\n";
    } else if (mode == ABI::Forward) {
      ss << "  enzyme::__enzyme_fwddiff(" << fn;
      for (size_t i = 0; i < out_shapes.size(); i++) {
        ss << ", enzyme_dup, ";
        ss << "&out_" << i << ", ";
        ss << "&dout_" << i;
      }
      if (tmpBuf != 0) {
        ss << ", enzyme_dup, &tmpBuf, &dtmpBuf";
      }
      for (size_t i = 0; i < in_shapes.size(); i++) {
        ss << ", enzyme_dup, ";
        ss << "&in_" << i << ", ";
        ss << "&din_" << i;
      }
      ss << ");\n";
    } else if (mode == ABI::Augmented) {
      // outs, tapeout
      // ins
      ss << "  std::size_t tapesize = enzyme::__enzyme_augmentsize(" << fn;
      for (size_t i = 0; i < out_shapes.size(); i++) {
        ss << ", enzyme_dup";
      }
      if (tmpBuf != 0) {
        ss << ", enzyme_dup";
      }
      for (size_t i = 0; i < in_shapes.size(); i++) {
        ss << ", enzyme_dup";
      }
      ss << ");\n";
      ss << "  enzyme::__enzyme_augmentfwd<void*>(" << fn
         << ", enzyme_allocated, tapesize, enzyme_tape, &tape";
      for (size_t i = 0; i < out_shapes.size(); i++) {
        ss << ", enzyme_dup, &out_" << i << ", nullptr";
      }
      if (tmpBuf != 0) {
        ss << ", enzyme_dup, &tmpBuf, &dtmpBuf";
      }
      for (size_t i = 0; i < in_shapes.size(); i++) {
        ss << ", enzyme_dup, &in_" << i << ", nullptr";
      }
      ss << ");\n";
    } else if (mode == ABI::Reverse) {

      // d_ins
      // tape, d_out

      // og outputs, og inputs
      //     doutputs (in), dinputs (out)
      ss << "  std::size_t tapesize = enzyme::__enzyme_augmentsize(" << fn;
      for (size_t i = 0; i < out_shapes.size(); i++) {
        ss << ", enzyme_dup";
      }
      if (tmpBuf != 0) {
        ss << ", enzyme_dup";
      }
      for (size_t i = 0; i < in_shapes.size(); i++) {
        ss << ", enzyme_dup";
      }
      ss << ");\n";
      for (size_t i = 0; i < in_shapes.size(); i++) {
        ss << "  din_" << i << " = (" << in_names[i] << ")0;\n";
      }
      ss << "  enzyme::__enzyme_reverse<void>(" << fn
         << ", enzyme_allocated, tapesize, enzyme_tape, &tape";
      for (size_t i = 0; i < out_shapes.size(); i++) {
        ss << ", enzyme_dup, nullptr, &dout_" << i;
      }
      if (tmpBuf != 0) {
        ss << ", enzyme_dup, &tmpBuf, &dtmpBuf";
      }
      for (size_t i = 0; i < in_shapes.size(); i++) {
        ss << ", enzyme_dup, nullptr, &din_" << i;
      }
      ss << ");\n";
      ss << "prevent_stores(";
      for (size_t i = 0; i < out_shapes.size(); i++) {
        if (i != 0)
          ss << ", ";
        ss << "(void*)&dout_" << i;
      }
      ss << ");\n";
    } else if (mode == ABI::Tape) {
      // outs, tapeout
      // ins
      ss << "  std::size_t tapesize = enzyme::__enzyme_augmentsize(" << fn;
      for (size_t i = 0; i < out_shapes.size(); i++) {
        ss << ", enzyme_dup";
      }
      if (tmpBuf != 0) {
        ss << ", enzyme_dup";
      }
      for (size_t i = 0; i < in_shapes.size(); i++) {
        ss << ", enzyme_dup";
      }
      ss << ");\n";
      ss << "  return tapesize;\n";
    } else {
      assert(0 && "unhandled mode");
    }
    ss << "}\n";

    llvm::SmallVector<std::string> pyargv_strs;
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
      pyargv_strs.emplace_back(argv);
#if PY_VERSION_HEX < 0x03000000
      free(argv);
#else
      // should not free py3+
#endif
    }

    auto mod = GetLLVMFromJob("/enzyme_call/source.cpp", ss.str(), /*cpp*/ true,
                              pyargv_strs, llvm_ctx.get(), std::move(linkMod));
    if (!mod) {
      llvm::errs() << "Source:\n" << ss.str() << "\n";
      throw pybind11::value_error("failed to compile C++");
    }
    return std::make_tuple(std::move(mod), std::move(llvm_ctx), out_off,
                           tmpBuf);
  }

  static std::pair<size_t, size_t>
  tapeAndTempSize(std::string fn, llvm::StringRef source,
                  llvm::ArrayRef<llvm::SmallVector<int64_t>> out_shapes,
                  llvm::ArrayRef<std::string> out_names,
                  llvm::ArrayRef<llvm::SmallVector<int64_t>> in_shapes,
                  llvm::ArrayRef<std::string> in_names, PyObject *pyargv,
                  Language lang, bool xla_runtime,
                  const std::string &pass_pipeline) {
    auto mode = ABI::Tape;
    auto [mod, llvm_ctx, num_out, tmpBuf] =
        createLLVMMod(fn, source, out_shapes, out_names, in_shapes, in_names,
                      pyargv, mode, lang, xla_runtime, pass_pipeline);
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
      auto local_executable = compile_mhlo_to_llvm_with_xla(
          source, llvm_ir, xla_runtime, pass_pipeline);
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

    auto [mod, llvm_ctx, num_out, tmpBuf] =
        createLLVMMod(fn, source, out_shapes, out_names, in_shapes, in_names,
                      pyargv, mode, lang, xla_runtime, pass_pipeline);

    if (!JIT) {
      DL = std::make_unique<llvm::DataLayout>(mod->getDataLayoutStr());
      auto tJIT =
          llvm::orc::LLJITBuilder()
              .setDataLayout(*DL.get())
              .setLinkProcessSymbolsByDefault(true)
              .setObjectLinkingLayerCreator(
                  [](llvm::orc::ExecutionSession &ES, const llvm::Triple &OLL)
                      -> llvm::Expected<
                          std::unique_ptr<llvm::orc::ObjectLayer>> {
                    auto obj = std::make_unique<
                        llvm::orc::RTDyldObjectLinkingLayer>(ES, []() {
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
        throw pybind11::value_error("failed to create jit");
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

void Callback(void *out, void **ins) {
  int64_t identifier = *reinterpret_cast<int64_t *>(ins[0]);
  CpuKernel *kernel = CpuKernel::get(identifier);
  if (!kernel) {
    if (identifier == CpuKernel::UNKNOWN_PLATFORM) {
      throw pybind11::value_error(
          "Unknown platform callback could not be executed");
    }
    // TODO: find a way to fail more gracefully.
    llvm::report_fatal_error("couldn't find enzyme kernel");
  }
  kernel->call(out, ins + 1);
}

extern "C" void RegisterEnzymeXLAGPUHandler();
extern "C" void RegisterEnzymeXLACPUHandler();

PYBIND11_MODULE(enzyme_call, m) {
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
  mlir::memref::registerResolveShapedTypeResultDims();
  mlir::registerLinalgPasses();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerConversionPasses();
  mlir::bufferization::registerBufferizationPasses();
  mlir::registerAsyncPasses();
  mlir::arith::registerArithPasses();
  mlir::memref::registerMemRefPasses();
  mlir::registerenzymePasses();
  mlir::enzyme::registerenzymexlaPasses();
  mlir::enzyme::registerGenerateApplyPatternsPass();
  mlir::enzyme::registerRemoveTransformPass();
  mlir::stablehlo::registerPasses();

  pybind11::enum_<Language>(m, "Language")
      .value("CPP", Language::CPP)
      .value("LLVM", Language::LLVM)
      .value("MHLO", Language::MHLO);

  pybind11::enum_<ABI>(m, "ABI")
      .value("Primal", ABI::Primal)
      .value("Forward", ABI::Forward)
      .value("Augmented", ABI::Augmented)
      .value("Reverse", ABI::Reverse)
      .value("Tape", ABI::Tape);

  m.def("create_enzyme_kernel",
        [](const std::string &source, const std::string &fn,
           const pybind11::list &py_out_shapes,
           const pybind11::list &py_in_shapes, pybind11::object pyargv,
           ABI mode, Language lang, bool xla_runtime,
           const std::string &pass_pipeline,
           const std::string &platform) -> std::tuple<size_t, size_t> {
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
           const std::string &fn, const pybind11::list &py_out_shapes,
           const pybind11::list &py_in_shapes, pybind11::object pyargv,
           Language lang, bool xla_runtime, const std::string &pass_pipeline) {
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

          std::error_code EC;
          llvm::raw_fd_ostream ostream(outfile, EC);

          auto [mod, llvm_ctx, num_out, tmpBuf] = CpuKernel::createLLVMMod(
              fn, source, out_shapes, out_types, in_shapes, in_types,
              pyargv.ptr(), ABI::Primal, lang, xla_runtime, pass_pipeline);

          ostream << *mod;
          ostream.close();
          return;
        });

  m.def("tape_and_tmp_size",
        [](const std::string &source, const std::string &fn,
           const pybind11::list &py_out_shapes,
           const pybind11::list &py_in_shapes, pybind11::object pyargv,
           Language lang, bool xla_runtime,
           const std::string &pass_pipeline) -> std::pair<size_t, size_t> {
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
          return CpuKernel::tapeAndTempSize(
              fn, source, out_shapes, out_types, in_shapes, in_types,
              pyargv.ptr(), (Language)lang, xla_runtime, pass_pipeline);
        });

  m.def("get_callback", []() {
    return pybind11::capsule(reinterpret_cast<void *>(&Callback),
                             "xla._CUSTOM_CALL_TARGET");
  });

  m.def("optimize_module",
        [](MlirModule cmod, const std::string &pass_pipeline) {
          run_pass_pipeline(unwrap(cmod), pass_pipeline);
        });
  m.def("run_pass_pipeline",
        [](pybind11::object pyoldsyms, const std::string &mlir,
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
          return run_pass_pipeline(oldsyms, mlir, pass_pipeline);
        });

  m.def("register_enzymexla_cpu_handler",
        []() { RegisterEnzymeXLACPUHandler(); });

  m.def("register_enzymexla_gpu_handler",
        []() { RegisterEnzymeXLAGPUHandler(); });

  m.def("compile_mhlo_to_llvm_with_xla",
        [](const std::string &mhlo_text, bool xla_runtime,
           const std::string &pass_pipeline) {
          std::string llvm_ir;
          compile_mhlo_to_llvm_with_xla(mhlo_text, llvm_ir, xla_runtime,
                                        pass_pipeline);
          return llvm_ir;
        });
}
