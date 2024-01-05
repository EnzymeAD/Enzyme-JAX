#pragma once
#include "xla/client/local_client.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
// Compile an MHLO module given as a string to LLVM IR using XLA.
std::unique_ptr<xla::LocalExecutable>
compile_mhlo_to_llvm_with_xla(llvm::StringRef mhlo_text, std::string &output,
                              bool xla_runtime);
