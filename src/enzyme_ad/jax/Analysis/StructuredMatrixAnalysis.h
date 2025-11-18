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
  LowerTriangular,
  Diagonal,
  Bidiagonal,
  Tridiagonal,
  Empty,
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
        upperBandwidth(upperBandwidth) {}

  static StructuredSparsityPattern meet(const StructuredSparsityPattern &lhs,
                                        const StructuredSparsityPattern &rhs) {
    if (lhs.kind == StructuredSparsityKind::Empty ||
        rhs.kind == StructuredSparsityKind::Empty)
      return StructuredSparsityPattern(StructuredSparsityKind::Empty);
    if (lhs.kind == StructuredSparsityKind::Unknown)
      return rhs;
    if (rhs.kind == StructuredSparsityKind::Unknown)
      return lhs;

    if (lhs.kind == StructuredSparsityKind::Band &&
        rhs.kind == StructuredSparsityKind::Band) {
      return StructuredSparsityPattern(
          std::min(lhs.lowerBandwidth, rhs.lowerBandwidth),
          std::min(lhs.upperBandwidth, rhs.upperBandwidth));
    }

    return lhs <= rhs ? lhs : rhs;
  }

  static StructuredSparsityPattern join(const StructuredSparsityPattern &lhs,
                                        const StructuredSparsityPattern &rhs) {
    if (lhs.kind == StructuredSparsityKind::Unknown ||
        rhs.kind == StructuredSparsityKind::Unknown)
      return StructuredSparsityPattern(StructuredSparsityKind::Unknown);
    if (lhs.kind == StructuredSparsityKind::Empty)
      return rhs;
    if (rhs.kind == StructuredSparsityKind::Empty)
      return lhs;

    if (lhs.kind == StructuredSparsityKind::Band &&
        rhs.kind == StructuredSparsityKind::Band) {
      return StructuredSparsityPattern(
          std::max(lhs.lowerBandwidth, rhs.lowerBandwidth),
          std::max(lhs.upperBandwidth, rhs.upperBandwidth));
    }

    return StructuredSparsityPattern(StructuredSparsityKind::Dense);
  }

  bool operator==(const StructuredSparsityPattern &other) const {}

  bool operator<=(const StructuredSparsityPattern &other) const {
    if (kind == StructuredSparsityKind::Empty)
      return true;

    if (other.kind == StructuredSparsityKind::Unknown)
      return true;

    if (other.kind == StructuredSparsityKind::Empty)
      return kind == StructuredSparsityKind::Empty;

    if (kind == StructuredSparsityKind::Unknown)
      return other.kind == StructuredSparsityKind::Unknown;

    if (kind == other.kind) {
      if (kind == StructuredSparsityKind::Band) {
        return lowerBandwidth <= other.lowerBandwidth &&
               upperBandwidth <= other.upperBandwidth;
      }
      return true;
    }

    if (kind == StructuredSparsityKind::Diagonal) {
      return other.kind != StructuredSparsityKind::Empty;
    }

    if (kind == StructuredSparsityKind::Bidiagonal) {
      return other.kind == StructuredSparsityKind::Tridiagonal ||
             other.kind == StructuredSparsityKind::Band ||
             other.kind == StructuredSparsityKind::UpperTriangular ||
             other.kind == StructuredSparsityKind::Dense;
    }

    if (kind == StructuredSparsityKind::Tridiagonal) {
      return other.kind == StructuredSparsityKind::Band ||
             other.kind == StructuredSparsityKind::Dense;
    }

    if (kind == StructuredSparsityKind::UpperTriangular ||
        kind == StructuredSparsityKind::LowerTriangular) {
      if (other.kind == StructuredSparsityKind::Dense)
        return true;
      if (other.kind == StructuredSparsityKind::Band) {
        if (kind == StructuredSparsityKind::UpperTriangular) {
          return other.lowerBandwidth == 0;
        } else {
          return other.upperBandwidth == 0;
        }
      }
      return false;
    }

    if (kind == StructuredSparsityKind::Band) {
      return other.kind == StructuredSparsityKind::Dense;
    }

    if (kind == StructuredSparsityKind::Dense) {
      return other.kind == StructuredSparsityKind::Dense ||
             other.kind == StructuredSparsityKind::Unknown;
    }

    return false;
  }

private:
  void initializeBandwidths();

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

  // partial ordering
  static ValueProperties meet(const ValueProperties &lhs,
                              const ValueProperties &rhs) {
    return ValueProperties(lhs.flags & rhs.flags);
  }

  static ValueProperties join(const ValueProperties &lhs,
                              const ValueProperties &rhs) {
    return ValueProperties(lhs.flags | rhs.flags);
  }

  bool operator==(const ValueProperties &other) const {
    return flags == other.flags;
  }

  bool operator<=(const ValueProperties &other) const {
    return (flags & other.flags) == flags;
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

  bool operator==(const StructuredMatrixType &other) const {
    return sparsityPattern == other.sparsityPattern &&
           valueProperties == other.valueProperties;
  }

private:
  StructuredSparsityPattern sparsityPattern;
  ValueProperties valueProperties;
};

} // namespace structure_analysis
} // namespace mlir
