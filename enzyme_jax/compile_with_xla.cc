//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>
#include <vector>

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
absl::StatusOr<std::string> compile_mhlo_to_llvm_with_xla(
    const std::string &mhlo_text) {
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
  if (!local_client_or_error.ok()) return local_client_or_error.status();
  xla::LocalClient *local_client = local_client_or_error.value();
  xla::ExecutableBuildOptions build_options;
  build_options.mutable_debug_options()->set_xla_embed_ir_in_executable(true);
  absl::StatusOr<std::vector<std::unique_ptr<xla::LocalExecutable>>>
      local_executables =
          local_client->Compile(xla_computation, shape_pointers, build_options);
  if (!local_executables.ok()) return local_executables.status();

  // Extract the LLVM IR stored in the executable.
  xla::LocalExecutable &local_executable = *local_executables.value()[0];
  auto *cpu_executable =
      static_cast<xla::cpu::CpuExecutable *>(local_executable.executable());
  const std::string &llvm_ir = cpu_executable->ir_module_string();
  return llvm_ir;
}
