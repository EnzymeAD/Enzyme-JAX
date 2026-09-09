#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Runtime/jit/jit.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#define DEBUG_TYPE "map-symbol"

namespace mlir::enzyme {
#define GEN_PASS_DEF_MAPSYMBOLPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace mlir::enzyme

using namespace mlir;
using namespace mlir::enzyme;

namespace {
struct MapSymbolPass : public enzyme::impl::MapSymbolPassBase<MapSymbolPass> {
  using MapSymbolPassBase::MapSymbolPassBase;

  void runOnOperation() override {
    enzymexla::InitJIT();

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

      auto err = enzymexla::MapSymbol(name.c_str(), addr);
      if (err) {
        llvm::errs() << "Failed to register symbol: " << name << " - "
                     << llvm::toString(std::move(err)) << "\n";
        return signalPassFailure();
      }
    }
  }
};
} // namespace
