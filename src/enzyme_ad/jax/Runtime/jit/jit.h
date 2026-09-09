#include <memory>

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/CoreContainers.h"
#include "llvm/Support/Error.h"

static std::unique_ptr<llvm::orc::LLJIT> JIT = nullptr;
static llvm::orc::SymbolMap MappedSymbols;

namespace enzymexla {

bool init_jit();
llvm::Error map_symbol(const char *name, void *symbol);
llvm::Expected<void *> lookup_symbol(const char *name);

} // namespace enzymexla
