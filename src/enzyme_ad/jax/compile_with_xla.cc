#define protected public
#include "xla/service/service.h"
#undef protected

#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/local_service_utils.h"

#include "absl/status/statusor.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "xla/client/client_library.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/xla_computation.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/printer.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "xla/translate/mhlo_to_hlo/type_to_shape.h"

#include "pybind11/pybind11.h"

#include "compile_with_xla.h"

// Compile an MHLO module given as a string to LLVM IR using XLA.
std::unique_ptr<xla::LocalExecutable>
compile_mhlo_to_llvm_with_xla(llvm::StringRef mhlo_text, std::string &output) {
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

  for (auto &computation :
       *hlo_proto.mutable_hlo_module()->mutable_computations()) {
    if (computation.id() != hlo_proto.hlo_module().entry_computation_id())
      continue;
    // Assume root is the last instruction.
    xla::HloInstructionProto &instruction =
        *computation.mutable_instructions()->rbegin();
    xla::cpu::BackendConfig backend_config;
    backend_config.ParseFromString(instruction.backend_config());
    backend_config.Clear();
    instruction.set_backend_config(backend_config.SerializeAsString());
    break;
  }

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
  // build_options.mutable_debug_options()->set_xla_cpu_use_xla_runtime(true);

  if (build_options.device_ordinal() == -1) {
    build_options.set_device_ordinal(local_client->default_device_ordinal());
  }

  absl::StatusOr<std::unique_ptr<xla::HloModuleConfig>> module_config_or_error =
      xla::GetHloModuleConfig(
          xla_computation, shape_pointers, build_options,
          /*(serice) options=*/&local_client->local_service()->options_,
          local_client->mutable_backend());
  if (!module_config_or_error.ok()) {
    throw pybind11::value_error(module_config_or_error.status().ToString());
  }
  module_config_or_error.value()->set_intra_op_parallelism_threads(1);

  auto executor = local_client->mutable_backend()->stream_executor(
      build_options.device_ordinal());
  if (!executor.ok()) {
    throw pybind11::value_error(executor.status().ToString());
  }

  auto executable = local_client->local_service()->BuildExecutable(
      xla_computation.proto(), std::move(module_config_or_error.value()),
      local_client->mutable_backend(), executor.value(),
      {build_options.device_allocator(), build_options.compile_thread_pool(),
       build_options.layout_canonicalization_callback()},
      build_options.run_backend_only());
  if (!executable.ok()) {
    throw pybind11::value_error(executable.status().ToString());
  }

  auto local_executable = std::make_unique<xla::LocalExecutable>(
      std::move(executable.value()),
      local_client->local_service()->mutable_backend(), build_options);

  auto *cpu_executable =
      static_cast<xla::cpu::CpuExecutable *>(local_executable->executable());

  output = cpu_executable->ir_module_string();
  return std::move(local_executable);
}
