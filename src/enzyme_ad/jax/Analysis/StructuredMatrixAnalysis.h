#include <algorithm>
#include <cstdint>

namespace mlir {
namespace structure_analysis {

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
      : kind(StructuredSparsityKind::Unknown), lowerBandwidth(0),
        upperBandwidth(0) {}

  explicit StructuredSparsityPattern(StructuredSparsityKind kind)
      : kind(kind), lowerBandwidth(-1), upperBandwidth(-1) {
    initializeBandwidths();
  }

  StructuredSparsityPattern(int64_t lowerBandwidth, int64_t upperBandwidth)
      : kind(StructuredSparsityKind::Band), lowerBandwidth(lowerBandwidth),
        upperBandwidth(upperBandwidth) {
    refineKind();
  }

  // most precise of lhs and rhs
  static StructuredSparsityPattern meet(const StructuredSparsityPattern &lhs,
                                        const StructuredSparsityPattern &rhs) {
    if (lhs.kind == StructuredSparsityKind::Empty ||
        rhs.kind == StructuredSparsityKind::Empty)
      return StructuredSparsityPattern(StructuredSparsityKind::Empty);

    if (lhs.kind == StructuredSparsityKind::Unknown)
      return rhs;
    if (rhs.kind == StructuredSparsityKind::Unknown)
      return lhs;

    // for all other cases, we take the min of the bandwidths and refine
    auto lb = std::min(lhs.lowerBandwidth, rhs.lowerBandwidth);
    auto ub = std::min(lhs.upperBandwidth, rhs.upperBandwidth);
    auto newPattern = StructuredSparsityPattern(lb, ub);
    newPattern.refineKind();
    return newPattern;
  }

  // least precise of lhs and rhs
  static StructuredSparsityPattern join(const StructuredSparsityPattern &lhs,
                                        const StructuredSparsityPattern &rhs) {
    if (lhs.kind == StructuredSparsityKind::Empty)
      return rhs;
    if (rhs.kind == StructuredSparsityKind::Empty)
      return lhs;

    if (lhs.kind == StructuredSparsityKind::Unknown ||
        rhs.kind == StructuredSparsityKind::Unknown)
      return StructuredSparsityPattern(StructuredSparsityKind::Unknown);

    auto lb = std::max(lhs.lowerBandwidth, rhs.lowerBandwidth);
    auto ub = std::max(lhs.upperBandwidth, rhs.upperBandwidth);
    auto newPattern = StructuredSparsityPattern(lb, ub);
    newPattern.refineKind();
    return newPattern;
  }

private:
  void initializeBandwidths();
  void refineKind();

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
};

class ValueProperties {
public:
  ValueProperties() = default;
  explicit ValueProperties(uint32_t flags) : flags(flags) {}

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

  uint32_t getFlags() const { return flags; }

  static ValueProperties meet(const ValueProperties &lhs,
                              const ValueProperties &rhs) {
    return ValueProperties(lhs.flags & rhs.flags);
  }

  static ValueProperties join(const ValueProperties &lhs,
                              const ValueProperties &rhs) {
    return ValueProperties(lhs.flags | rhs.flags);
  }

private:
  uint32_t flags;
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

  const StructuredSparsityPattern &getSparsityPattern() const {
    return sparsityPattern;
  }
  const ValueProperties &getProperties() const { return valueProperties; }

  // partial ordering
  static StructuredMatrixType meet(const StructuredMatrixType &lhs,
                                   const StructuredMatrixType &rhs) {
    return StructuredMatrixType(
        StructuredSparsityPattern::meet(lhs.sparsityPattern,
                                        rhs.sparsityPattern),
        ValueProperties::meet(lhs.valueProperties, rhs.valueProperties));
  }

  static StructuredMatrixType join(const StructuredMatrixType &lhs,
                                   const StructuredMatrixType &rhs) {
    return StructuredMatrixType(
        StructuredSparsityPattern::join(lhs.sparsityPattern,
                                        rhs.sparsityPattern),
        ValueProperties::join(lhs.valueProperties, rhs.valueProperties));
  }

private:
  StructuredSparsityPattern sparsityPattern;
  ValueProperties valueProperties;
};

} // namespace structure_analysis
} // namespace mlir
