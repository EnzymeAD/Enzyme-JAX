#include "src/enzyme_ad/jax/Passes/Passes.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#define DEBUG_TYPE "jit-map-symbol"

namespace mlir::enzyme {
#define GEN_PASS_DEF_MAPSYMBOLPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace mlir::enzyme

using namespace mlir;
using namespace mlir::enzyme;

// from LowerJIT.cpp
extern "C" void EnzymeJaXMapSymbol(const char *name, void *symbol);
extern "C" int EnzymeJaXLookupSymbol(const char *name, void **symbol);

namespace {
struct MapSymbolPass : public enzyme::impl::MapSymbolPassBase<MapSymbolPass> {
  using MapSymbolPassBase::MapSymbolPassBase;

  void runOnOperation() override {
    for (const std::string &symbol : symbols) {
      std::string name = symbol;
      void *addr = nullptr;
      uint64_t value;

      const size_t separator = symbol.find('=');
      if (separator != std::string::npos) {
        const std::string rhs = symbol.substr(separator + 1);

        try {
          value = std::stoull(rhs, nullptr, rhs.rfind("0x", 0) == 0 ? 16 : 10);
        } catch (const std::invalid_argument &) {
          llvm::errs() << "Invalid address (" << rhs << ") for symbol (" << name
                       << ")\n";
          return signalPassFailure();
        } catch (const std::out_of_range &) {
          llvm::errs() << "Value out of range for symbol mapping: " << rhs
                       << "\n";
          return signalPassFailure();
        }
        name = symbol.substr(0, separator);
        addr = reinterpret_cast<void *>(static_cast<uintptr_t>(value));
      }

      EnzymeJaXMapSymbol(name.c_str(), addr);

      void *lookup_addr;
      int found_addr = EnzymeJaXLookupSymbol(name.c_str(), &lookup_addr);
      if (found_addr != 0) {
        llvm::errs() << "`" << name << "` symbol not mapped\n";
        return signalPassFailure();
      }
    }
  }
};
} // namespace
