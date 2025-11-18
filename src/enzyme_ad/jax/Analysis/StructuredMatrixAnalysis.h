#pragma once

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

#include <algorithm>
#include <cstdint>

namespace mlir {
namespace structure_analysis {

namespace utils {

static bool isOne(APInt v) { return v.isOne(); }
static bool isOne(APFloat v) { return v.isExactlyValue(1.0); }
static bool isOne(Attribute v) {
  if (auto intAttr = dyn_cast<IntegerAttr>(v))
    return isOne(intAttr.getValue());
  if (auto floatAttr = dyn_cast<FloatAttr>(v))
    return isOne(floatAttr.getValue());
  return false;
}

static bool areEqual(APInt a, APInt b) { return a == b; }
static bool areEqual(APFloat a, APFloat b) {
  return a.compare(b) == llvm::APFloat::cmpEqual;
}

} // namespace utils

//===----------------------------------------------------------------------===//
// Structured Sparsity Pattern Implementation
//===----------------------------------------------------------------------===//

enum class StructuredSparsityKind {
  Unknown,
  Dense,
  Band,
  UpperTriangular,
  UpperBidiagonal,
  LowerTriangular,
  LowerBidiagonal,
  Tridiagonal,
  Diagonal,
  Empty, // doesn't really mean anything, but we need it for bottom element
};

// TODO: currently only legal negative value is -1, which means "unknown"
// we should support negative bandwidths
class StructuredSparsityPattern {
public:
  StructuredSparsityPattern()
      : kind(StructuredSparsityKind::Unknown), lowerBandwidth(-1),
        upperBandwidth(-1) {}

  explicit StructuredSparsityPattern(StructuredSparsityKind kind)
      : kind(kind), lowerBandwidth(-1), upperBandwidth(-1) {
    initializeBandwidths();
  }

  StructuredSparsityPattern(Value v);

  StructuredSparsityPattern(int64_t lowerBandwidth, int64_t upperBandwidth)
      : kind(StructuredSparsityKind::Band), lowerBandwidth(lowerBandwidth),
        upperBandwidth(upperBandwidth) {
    refineKind();
  }

  static StructuredSparsityPattern meet(const StructuredSparsityPattern &lhs,
                                        const StructuredSparsityPattern &rhs);

  static StructuredSparsityPattern join(const StructuredSparsityPattern &lhs,
                                        const StructuredSparsityPattern &rhs);

  bool operator==(const StructuredSparsityPattern &other) const {
    return kind == other.kind && lowerBandwidth == other.lowerBandwidth &&
           upperBandwidth == other.upperBandwidth;
  }

  void print(raw_ostream &os) const;
  raw_ostream &operator<<(raw_ostream &os) const {
    print(os);
    return os;
  }

private:
  void initializeBandwidths();
  void refineKind();

  void setUnknown() {
    kind = StructuredSparsityKind::Unknown;
    lowerBandwidth = -1;
    upperBandwidth = -1;
  }

  StructuredSparsityKind kind;
  int64_t lowerBandwidth;
  int64_t upperBandwidth;
};

//===----------------------------------------------------------------------===//
// Value Properties Implementation
//===----------------------------------------------------------------------===//

enum class ValueProperty {
  UnitDiagonal = 1 << 0,
  Symmetric = 1 << 1,
  Hermitian = 1 << 2,
  BroadcastedScalar = 1 << 3,
};

class ValueProperties {
public:
  ValueProperties() = default;
  explicit ValueProperties(uint32_t flags) : flags(flags) {}

  ValueProperties(Value v);

  void set(ValueProperty property) { flags |= static_cast<uint32_t>(property); }
  void clear(ValueProperty property) {
    flags &= ~static_cast<uint32_t>(property);
  }
  bool has(ValueProperty property) const {
    return flags & static_cast<uint32_t>(property);
  }

  bool hasUnitDiagonal() const { return has(ValueProperty::UnitDiagonal); }
  bool isSymmetric() const { return has(ValueProperty::Symmetric); }
  bool isHermitian() const { return has(ValueProperty::Hermitian); }
  bool isBroadcastedScalar() const {
    return has(ValueProperty::BroadcastedScalar);
  }

  void print(raw_ostream &os) const;
  raw_ostream &operator<<(raw_ostream &os) const {
    print(os);
    return os;
  }

  uint32_t getFlags() const { return flags; }
  void setFlags(uint32_t f) { flags = f; }

  static ValueProperties meet(const ValueProperties &lhs,
                              const ValueProperties &rhs);

  static ValueProperties join(const ValueProperties &lhs,
                              const ValueProperties &rhs);

  bool operator==(const ValueProperties &other) const {
    return flags == other.flags;
  }

private:
  static ValueProperties getPropertiesFromDenseAttr(DenseElementsAttr attr);

  static bool isUnitDiagonal(DenseElementsAttr attr, int64_t nrows,
                             int64_t ncols);
  static std::tuple<int64_t, int64_t>
  isSymmetricOrHermitian(DenseElementsAttr, int64_t nrows, int64_t ncols);

  uint32_t flags = 0;
};

//===----------------------------------------------------------------------===//
// Structured Matrix Type
//===----------------------------------------------------------------------===//

class StructuredMatrixType {
public:
  StructuredMatrixType() = default;
  StructuredMatrixType(StructuredSparsityPattern sparsityPattern,
                       ValueProperties valueProperties)
      : sparsityPattern(sparsityPattern), valueProperties(valueProperties) {}

  StructuredMatrixType(Value v)
      : StructuredMatrixType(StructuredSparsityPattern(v), ValueProperties(v)) {
  }

  const StructuredSparsityPattern &getSparsityPattern() const {
    return sparsityPattern;
  }
  const ValueProperties &getProperties() const { return valueProperties; }

  static StructuredMatrixType meet(const StructuredMatrixType &lhs,
                                   const StructuredMatrixType &rhs);

  static StructuredMatrixType join(const StructuredMatrixType &lhs,
                                   const StructuredMatrixType &rhs);

  bool operator==(const StructuredMatrixType &other) const {
    return sparsityPattern == other.sparsityPattern &&
           valueProperties == other.valueProperties;
  }

  void print(raw_ostream &os) const;
  raw_ostream &operator<<(raw_ostream &os) const {
    print(os);
    return os;
  }

  // TODO: propagation rules probably goes in here

  // TODO: implement queries that check both the sparsity pattern and value
  // properties and return specific matrix kinds

private:
  StructuredSparsityPattern sparsityPattern;
  ValueProperties valueProperties;
};

//===----------------------------------------------------------------------===//
// Lattice Element
//===----------------------------------------------------------------------===//

class StructuredMatrixLattice : public dataflow::AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;

  StructuredMatrixLattice(Value v)
      : AbstractSparseLattice(v), value(StructuredMatrixType(v)) {}

  ChangeResult meet(const AbstractSparseLattice &rhs) override;
  ChangeResult meet(StructuredMatrixLattice rhs);

  ChangeResult join(const AbstractSparseLattice &rhs) override;
  ChangeResult join(StructuredMatrixLattice rhs);

  void print(raw_ostream &os) const override;
  raw_ostream &operator<<(raw_ostream &os) const {
    print(os);
    return os;
  }

  const StructuredMatrixType &getValue() const { return value; }
  void setValue(const StructuredMatrixType &v) { value = v; }

private:
  StructuredMatrixType value;
};

//===----------------------------------------------------------------------===//
// Dataflow Analysis
//===----------------------------------------------------------------------===//

class StructuredMatrixAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<StructuredMatrixLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  void setToEntryState(StructuredMatrixLattice *lattice) override;

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const StructuredMatrixLattice *> operands,
                 ArrayRef<StructuredMatrixLattice *> results) override;
};

//===----------------------------------------------------------------------===//
// Structure Originators
//===----------------------------------------------------------------------===//

} // namespace structure_analysis
} // namespace mlir
