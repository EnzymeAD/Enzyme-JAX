#ifndef PACT_ANALYSIS_PRODUCT_LATTICE_H
#define PACT_ANALYSIS_PRODUCT_LATTICE_H

#include "FactState.h"
#include "PipelineLevel.h"
#include "src/enzyme_ad/jax/Passes/Pact/PropertyScheme.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

template <> struct DenseMapInfo<mlir::enzyme::pact::Level> {
  static mlir::enzyme::pact::Level getEmptyKey() {
    return static_cast<mlir::enzyme::pact::Level>(-1);
  }
  static mlir::enzyme::pact::Level getTombstoneKey() {
    return static_cast<mlir::enzyme::pact::Level>(-2);
  }
  static unsigned getHashValue(mlir::enzyme::pact::Level l) {
    return DenseMapInfo<int>::getHashValue(static_cast<int>(l));
  }
  static bool isEqual(mlir::enzyme::pact::Level a,
                      mlir::enzyme::pact::Level b) {
    return a == b;
  }
};

} // namespace llvm

namespace mlir::enzyme::pact {

struct PropertyColumn {
  llvm::StringRef propertyKey;
  scheme::PropertyKind kind;
  Level meetLevel = Level::L2;
  llvm::DenseMap<Level, FactEntry> entries;
  llvm::SmallVector<std::pair<Level, FactState>> trace;

  PropertyColumn() = default;
  PropertyColumn(llvm::StringRef key, scheme::PropertyKind k, Level lvl)
      : propertyKey(key), kind(k), meetLevel(lvl) {}

  FactEntry &at(Level level);
  const FactEntry *get(Level level) const;
  void transition(Level level, FactState newState, llvm::StringRef why = "");
  FactState currentState() const;
};

struct ProductLattice {
  llvm::SmallVector<PropertyColumn> columns;

  PropertyColumn *findColumn(llvm::StringRef key);
  const PropertyColumn *findColumn(llvm::StringRef key) const;
  PropertyColumn &addColumn(llvm::StringRef key, scheme::PropertyKind kind,
                            Level meetLevel);

  bool allResolved() const;
  int countMismatches() const;
  llvm::SmallVector<PropertyColumn *> unresolvedColumns();
  bool hasBlocking() const;
};

} // namespace mlir::enzyme::pact

#endif // PACT_ANALYSIS_PRODUCT_LATTICE_H