#include <string>

#define protected public
#include "xla/service/service.h"
#undef protected

// Needed to access CompileXlaRuntimeCpuExecutable/etc
#define private public
#include "xla/service/cpu/cpu_compiler.h"
#undef private

#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/service/local_service_utils.h"

#include "absl/status/statusor.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "pybind11/pybind11.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/client/client_library.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/xla_computation.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/printer.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "xla/translate/mhlo_to_hlo/type_to_shape.h"

#include "compile_with_xla.h"

#include "TransformOps/TransformOps.h"

#include "pybind11/stl.h"

#include "RegistryUtils.h"

/// Returns an unused symbol in `module` for `oldSymbolName` by trying numeric
/// suffix in `lastUsedID`.
static mlir::StringAttr renameSymbol(llvm::StringRef oldSymName,
                                     unsigned &lastUsedID,
                                     std::set<std::string> &oldsym,
                                     mlir::MLIRContext *ctx) {
  using namespace llvm;
  using namespace mlir;
  SmallString<64> newSymName(oldSymName);
  newSymName.push_back('_');

  while (true) {
    auto possible = newSymName + Twine(++lastUsedID);
    if (!oldsym.count(possible.str())) {
      oldsym.insert(possible.str());
      return StringAttr::get(ctx, possible);
    }
  }
}

/// Checks if a symbol with the same name as `op` already exists in `source`.
/// If so, renames `op` and updates all its references in `target`.
static mlir::LogicalResult
updateSymbolAndAllUses(mlir::SymbolOpInterface op, mlir::ModuleOp target,
                       std::set<std::string> &oldsyms, unsigned &lastUsedID) {
  using namespace llvm;
  using namespace mlir;

  auto opName = op.getName().str();

  if (!oldsyms.count(opName)) {
    oldsyms.insert(opName);
    return success();
  }

  StringAttr newSymName =
      renameSymbol(opName, lastUsedID, oldsyms, target.getContext());

  if (failed(SymbolTable::replaceAllSymbolUses(op, newSymName, target)))
    return op.emitError("unable to update all symbol uses for ")
           << opName << " to " << newSymName;

  SymbolTable::setSymbolName(op, newSymName);
  return success();
}

void run_pass_pipeline(mlir::Operation *mod, const std::string &pass_pipeline) {
  using namespace llvm;
  using namespace mlir;

  mlir::DialectRegistry registry;
  prepareRegistry(registry);
  mod->getContext()->appendDialectRegistry(registry);

  mlir::PassManager pm(mod->getContext());
  std::string error_message;
  llvm::raw_string_ostream error_stream(error_message);
  mlir::LogicalResult result =
      mlir::parsePassPipeline(pass_pipeline, pm, error_stream);
  if (mlir::failed(result)) {
    throw pybind11::value_error(error_message);
  }

  DiagnosticEngine &engine = mod->getContext()->getDiagEngine();
  error_stream << "Pipeline failed:\n";
  DiagnosticEngine::HandlerID id =
      engine.registerHandler([&](Diagnostic &diag) -> LogicalResult {
        error_stream << diag << "\n";
        return failure();
      });
  if (!mlir::succeeded(pm.run(cast<mlir::ModuleOp>(mod)))) {
    throw pybind11::value_error(error_stream.str());
  }
}

std::pair<std::string, std::string>
run_pass_pipeline(const std::vector<std::string> &oldsym_vec,
                  const std::string &mlir, const std::string &pass_pipeline) {
  using namespace llvm;
  using namespace mlir;

  std::set<std::string> oldsyms(oldsym_vec.begin(), oldsym_vec.end());

  // Parse MLIR.
  mlir::DialectRegistry registry;
  prepareRegistry(registry);
  MLIRContext context(registry);
  mlir::ParserConfig parser_config(&context);
  mlir::OwningOpRef<mlir::ModuleOp> parsed_module =
      mlir::parseSourceString<mlir::ModuleOp>(mlir, parser_config);
  if (!parsed_module) {
    throw pybind11::value_error("Failed to parse module");
  }

  mlir::PassManager pm(&context);

  std::string error_message;
  llvm::raw_string_ostream error_stream(error_message);
  error_stream << "Failed to parse pipeline\n";
  mlir::LogicalResult result =
      mlir::parsePassPipeline(pass_pipeline, pm, error_stream);
  if (mlir::failed(result)) {
    throw pybind11::value_error(error_message);
  }

  DiagnosticEngine &engine = context.getDiagEngine();
  error_stream << "Pipeline failed:\n";
  DiagnosticEngine::HandlerID id =
      engine.registerHandler([&](Diagnostic &diag) -> LogicalResult {
        error_stream << diag << "\n";
        return failure();
      });
  if (!mlir::succeeded(pm.run(cast<mlir::ModuleOp>(*parsed_module)))) {
    throw pybind11::value_error(error_stream.str());
  }

  StringRef entryfn = "main";

  unsigned lastUsedID = 0;

  for (auto &op : *parsed_module->getBody()) {
    auto symbolOp = dyn_cast<SymbolOpInterface>(op);
    if (!symbolOp)
      continue;

    StringRef oldSymName = symbolOp.getName();

    if (failed(updateSymbolAndAllUses(symbolOp, *parsed_module, oldsyms,
                                      lastUsedID)))
      throw pybind11::value_error("failed to update all uses");

    StringRef newSymName = symbolOp.getName();
    if (oldSymName != newSymName) {
      if (oldSymName == entryfn) {
        entryfn = newSymName;
      }
    }
    if (newSymName == entryfn) {
      SymbolTable::setSymbolVisibility(&op, SymbolTable::Visibility::Private);
    }
  }

  std::string output;
  llvm::raw_string_ostream ss(output);
  parsed_module->getOperation()->print(
      ss, mlir::OpPrintingFlags().enableDebugInfo());

  return std::make_pair(entryfn.str(), ss.str());
}

absl::StatusOr<std::unique_ptr<xla::Executable>>
RunBackend(xla::cpu::CpuCompiler *self, std::unique_ptr<xla::HloModule> module,
           [[maybe_unused]] xla::se::StreamExecutor *stream_exec,
           const xla::Compiler::CompileOptions &options, bool xla_runtime) {

  std::unique_ptr<xla::cpu::CpuExecutable> cpu_executable;
  if (xla_runtime) {
    throw pybind11::value_error("xla_runtime deprecated upstream");
    // TF_ASSIGN_OR_RETURN(cpu_executable,
    //                    self->CompileXlaRuntimeCpuExecutable(std::move(module),
    //                                                         options.registry));
  } else {
    TF_ASSIGN_OR_RETURN(cpu_executable,
                        self->CompileCpuExecutable(std::move(module)));
  }

  return std::unique_ptr<xla::Executable>(std::move(cpu_executable));
}

absl::StatusOr<std::unique_ptr<xla::Executable>>
BuildExecutable(xla::Service *self, const xla::HloModuleProto &module_proto,
                std::unique_ptr<xla::HloModuleConfig> module_config,
                xla::Backend *backend, xla::se::StreamExecutor *executor,
                const xla::Compiler::CompileOptions &options,
                bool run_backend_only, bool xla_runtime) {

  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::HloModule> module,
                      xla::CreateModuleFromProto(module_proto, *module_config,
                                                 run_backend_only));
  xla::UpdateEntryComputationLayout(
      module.get(), std::bind(&xla::Compiler::DefaultDeviceShapeRepresentation,
                              backend->compiler(), std::placeholders::_1));
  // xla::DumpHloModuleIfEnabled(*module, xla::kBeforeOptimizationsDumpName);

  std::unique_ptr<xla::HloProto> hlo_proto_before_opt;
  if (!run_backend_only) {
    // Save proto state before optimizations if we want a snapshot.
    // When run_backend_only is enabled the post-optimization HLO will be the
    // same as the pre-optimization HLO.
    // if (xla::DumpingEnabledForHloModule(*module)) {
    //  hlo_proto_before_opt =
    //  std::make_unique<xla::HloProto>(MakeHloProto(*module));
    // }
    TF_ASSIGN_OR_RETURN(module, backend->compiler()->RunHloPasses(
                                    std::move(module), executor, options));
  }

  /*
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::Executable> executable,
      backend->compiler()->RunBackend(std::move(module), executor, options));
  */
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::Executable> executable,
                      RunBackend((xla::cpu::CpuCompiler *)backend->compiler(),
                                 std::move(module), executor, options,
                                 xla_runtime));

  const xla::BufferAssignmentProto *buffer_assignment_proto_after_opt =
      executable->buffer_assignment_proto();

  // If dumping is enabled RunBackend(...) will emit a hlo_proto in the
  // executable. This contains the buffer_assignment that is only available
  // after RunBackend(). If hlo_proto_before_opt is not null, then we replace
  // its buffer_assignment with the one from after_opt and then store it into
  // the executable.
  if (hlo_proto_before_opt != nullptr &&
      buffer_assignment_proto_after_opt != nullptr) {
    // CHECK(xla::DumpingEnabledForHloModule(executable->module()));
    *hlo_proto_before_opt->mutable_buffer_assignment() =
        std::move(*buffer_assignment_proto_after_opt);
    executable->set_hlo_proto(std::move(hlo_proto_before_opt));
  }
  return std::move(executable);
}

// Compile an MHLO module given as a string to LLVM IR using XLA.
std::unique_ptr<xla::LocalExecutable>
compile_mhlo_to_llvm_with_xla(llvm::StringRef mhlo_text, std::string &output,
                              bool xla_runtime,
                              const std::string &pass_pipeline) {
  // Parse MLIR.
  mlir::DialectRegistry registry;
  prepareRegistry(registry);
  mlir::MLIRContext context(registry);
  mlir::ParserConfig parser_config(&context);
  mlir::OwningOpRef<mlir::ModuleOp> parsed_module =
      mlir::parseSourceString<mlir::ModuleOp>(mhlo_text, parser_config);
  if (!parsed_module) {
    throw pybind11::value_error("Failed to parse module");
  }

  llvm::StringRef cur_pipeline = pass_pipeline;

  mlir::PassManager pm(&context);
  if (!xla_runtime) {
    pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  } else {
    std::string tofind = "stablehlo-legalize-to-hlo,";
    auto pos = llvm::StringRef(pass_pipeline).find(tofind);
    assert(pos != std::string::npos);
    auto pre = llvm::StringRef(pass_pipeline.data(), pos + tofind.size() - 1);
    cur_pipeline = llvm::StringRef(pass_pipeline.data() + pos + tofind.size(),
                                   pass_pipeline.size() - pos - tofind.size());

    std::string error_message;
    llvm::raw_string_ostream error_stream(error_message);
    error_stream << "Failed to parse pre stablehlo pipeline\n";
    mlir::LogicalResult result = mlir::parsePassPipeline(pre, pm, error_stream);
    if (mlir::failed(result)) {
      throw pybind11::value_error(error_message);
    }
  }
  if (!mlir::succeeded(pm.run(*parsed_module))) {
    throw pybind11::value_error("StableHLO => MHLO failed");
  }

  // Convert to XLA Computation.
  xla::HloProto hlo_proto;
  auto status = mlir::ConvertMlirHloToHlo(*parsed_module, &hlo_proto,
                                          /*use_tuple_args=*/false,
                                          /*return_tuple=*/false);

  if (!status.ok()) {
    throw pybind11::value_error(std::string(status.message()));
  }

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
  build_options.mutable_debug_options()->set_xla_cpu_use_thunk_runtime(false);

  build_options.mutable_debug_options()
      ->mutable_xla_backend_extra_options()
      ->emplace("xla_cpu_experimental_override_pipeline", cur_pipeline.str());

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

  xla::Compiler::CompileOptions opts = {
      build_options.device_allocator(), build_options.compile_thread_pool(),
      build_options.layout_canonicalization_callback()};
  auto executable =
      BuildExecutable(local_client->local_service(), xla_computation.proto(),
                      std::move(module_config_or_error.value()),
                      local_client->mutable_backend(), executor.value(), opts,
                      build_options.run_backend_only(), xla_runtime);
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
