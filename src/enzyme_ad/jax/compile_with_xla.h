#pragma once
#include "xla/client/local_client.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

#include <utility>

// Compile an MHLO module given as a string to LLVM IR using XLA.
std::unique_ptr<xla::LocalExecutable>
compile_mhlo_to_llvm_with_xla(llvm::StringRef mhlo_text, std::string &output,
                              bool xla_runtime,
                              const std::string &pass_pipeline);

std::pair<std::string, std::string>
run_pass_pipeline(const std::vector<std::string> &oldsyms,
                  const std::string &mlir, const std::string &pass_pipeline);

namespace mlir {
class Operation;
}
void run_pass_pipeline(mlir::Operation *mod, const std::string &pass_pipeline);
void prepareRegistry(mlir::DialectRegistry &registry);
