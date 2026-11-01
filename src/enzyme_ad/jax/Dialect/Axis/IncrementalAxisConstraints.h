#ifndef ENZYME_AD_JAX_DIALECT_AXIS_INCREMENTALAXISCONSTRAINTS_H
#define ENZYME_AD_JAX_DIALECT_AXIS_INCREMENTALAXISCONSTRAINTS_H

#include <atomic>
#include <cstdint>

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::enzyme::axis {

// A digit of some tensor dimension with a known extent, identified by a
// globally unique id.
class AxisSymbol {
public:
  AxisSymbol() = default;
  AxisSymbol(const AxisSymbol &) = default;
  AxisSymbol &operator=(const AxisSymbol &) = default;

  static AxisSymbol create(uint64_t extent) {
    return AxisSymbol(nextId.fetch_add(1, std::memory_order_relaxed), extent);
  }

  static AxisSymbol getEmptyKey() { return AxisSymbol(kEmptyId, 0); }
  static AxisSymbol getTombstoneKey() { return AxisSymbol(kTombstoneId, 0); }

  bool isNull() const { return id == kNullId; }
  uint64_t getId() const { return id; }
  uint64_t getExtent() const { return extent; }
  bool operator==(const AxisSymbol &other) const { return id == other.id; }

private:
  explicit AxisSymbol(uint64_t id, uint64_t extent) : id(id), extent(extent) {}

  static constexpr uint64_t kNullId = 0;
  static constexpr uint64_t kEmptyId = UINT64_MAX;
  static constexpr uint64_t kTombstoneId = UINT64_MAX - 1;
  inline static std::atomic<uint64_t> nextId{1};
  uint64_t id = kNullId;
  uint64_t extent;
};

} // namespace mlir::enzyme::axis

namespace llvm {

template <> struct DenseMapInfo<mlir::enzyme::axis::AxisSymbol> {
  static mlir::enzyme::axis::AxisSymbol getEmptyKey() {
    return mlir::enzyme::axis::AxisSymbol::getEmptyKey();
  }

  static mlir::enzyme::axis::AxisSymbol getTombstoneKey() {
    return mlir::enzyme::axis::AxisSymbol::getTombstoneKey();
  }

  static unsigned getHashValue(const mlir::enzyme::axis::AxisSymbol &k) {
    return DenseMapInfo<uint64_t>::getHashValue(k.getId());
  }

  static bool isEqual(const mlir::enzyme::axis::AxisSymbol &lhs,
                      const mlir::enzyme::axis::AxisSymbol &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

namespace mlir::enzyme::axis {

/**
 * The incremental solution of the axis-string constraints added so far.
 *
 * A string is a list of symbols, major-first, naming the digits of one tensor
 * dimension. Two kinds of constraint are added:
 * - a meet `a = b` identifies two strings of one dimension, merging their
 *   digits where the factorings align and factoring symbols as needed;
 * - an overlap `overlap(a ++ b ++ ...)` keeps the digits of the strings, and
 *   any future factoring of them, distinct.
 * A digit that can't satisfy both becomes local (L): it is never sharded.
 *
 * The solution is a union-find over symbols plus a factorization of each root
 * into finer symbols; `resolve` reads a string's current root digits. String
 * operations expect each string to be unique even after resolution, and the
 * two strings of a meet to have equal total extent.
 */
class IncrementalAxisConstraints {
public:
  using OverlapSet = llvm::DenseSet<AxisSymbol>;

  // Why two strings can't be unified over some stretch of their dimension.
  enum class MergeConflict {
    None,
    // A symbol on one side overlaps a symbol on the other.
    Overlap,
    // A symbol on both sides sits at different positions in each, as in
    // `x + x.T` for a square `x`.
    Order,
    // The factorings have no common refinement, as 2*3 against 3*2.
    Unaligned,
  };
  // One aligned stretch of two strings: `lhs` and `rhs` have equal extent.
  struct GroupPair {
    llvm::SmallVector<AxisSymbol> lhs, rhs;
    MergeConflict conflict;
  };
  // A stretch of the dimension: `extent` digits whose minor-most has stride
  // `stride`.
  struct ConflictRange {
    uint64_t stride;
    uint64_t extent;
    MergeConflict conflict;
  };

  // Strings.

  // Adds `a = b`: conflict-free stretches merge, and every symbol of a
  // conflicting stretch becomes local. An overlap stretch that is one aligned
  // pair still merges, into a local class.
  void meet(llvm::ArrayRef<AxisSymbol> a, llvm::ArrayRef<AxisSymbol> b);
  // Succeeds iff meet(a, b) would localize nothing. Changes nothing.
  LogicalResult checkMeetNoLocalize(llvm::ArrayRef<AxisSymbol> a,
                                    llvm::ArrayRef<AxisSymbol> b);
  // Merges every stretch of align(a, b) that has no conflict. A stretch with a
  // conflict keeps its symbols distinct and, unlike in meet, none is
  // localized.
  void partialMeetNoLocalize(llvm::ArrayRef<AxisSymbol> a,
                             llvm::ArrayRef<AxisSymbol> b);
  // meet(a, b) if it would localize nothing; otherwise changes nothing and
  // fails.
  LogicalResult strictMeetNoLocalize(llvm::ArrayRef<AxisSymbol> a,
                                     llvm::ArrayRef<AxisSymbol> b);
  // Adds an overlap among the given symbols: pairwise, not reflexive, and not
  // transitive across separate calls.
  void overlap(llvm::ArrayRef<AxisSymbol> symbols);
  /**
   * Factors both strings at the borders they share and pairs up the aligned
   * stretches, major-first. A symbol is cut only where one of its pieces is
   * conflict-free, and an order conflict leaves both strings whole. A symbol
   * on both sides pairs with itself, and any other extent-1 symbol pairs with
   * nothing. Merges and localizes nothing.
   */
  llvm::SmallVector<GroupPair> align(llvm::ArrayRef<AxisSymbol> a,
                                     llvm::ArrayRef<AxisSymbol> b);
  // The stretches that meet(a, b) would localize. Changes nothing.
  llvm::SmallVector<ConflictRange> conflicts(llvm::ArrayRef<AxisSymbol> a,
                                             llvm::ArrayRef<AxisSymbol> b);

  // Digits. Each operand is a root symbol (its own resolution).

  // Factors `symbol` into fresh pieces of the given extents, major-first, and
  // returns them.
  llvm::SmallVector<AxisSymbol> factor(AxisSymbol symbol,
                                       llvm::ArrayRef<uint64_t> extents);
  // Factors `symbol` into existing root symbols, major-first. The pieces
  // inherit the symbol's overlaps and overlap each other, since merging two
  // digits of one axis would be a diagonal.
  void factorInto(AxisSymbol symbol, llvm::ArrayRef<AxisSymbol> pieces);
  // Merges two root symbols into one class, which keeps both overlap sets and
  // is local if either was or if they overlap.
  void meetSymbols(AxisSymbol a, AxisSymbol b);
  // Marks the symbols local. They still merge normally, and every symbol
  // merged with or factored from them stays local.
  void localize(llvm::ArrayRef<AxisSymbol> symbols);

  // Queries.

  // The current root digits of a symbol or string.
  llvm::SmallVector<AxisSymbol> resolve(AxisSymbol symbol);
  llvm::SmallVector<AxisSymbol> resolve(llvm::ArrayRef<AxisSymbol> symbols);
  // The root symbols that overlap some root digit of a symbol or string.
  OverlapSet overlapping(AxisSymbol symbol);
  OverlapSet overlapping(llvm::ArrayRef<AxisSymbol> symbols);
  // Whether some root digit of the symbol is local.
  bool isLocal(AxisSymbol symbol);

private:
  llvm::EquivalenceClasses<AxisSymbol> symbolUnion;
  llvm::DenseMap<AxisSymbol, llvm::SmallVector<AxisSymbol>> factorizations;
  // Keyed by root symbol; an entry is dropped once its symbol stops being a
  // root.
  llvm::DenseMap<AxisSymbol, OverlapSet> overlappingSymbols;
  // Keyed by root symbol, like overlappingSymbols.
  llvm::DenseSet<AxisSymbol> localSymbols;

  // Merges an aligned pair, at least one side of which is one symbol.
  void mergeAlignedPair(const GroupPair &pair);
  void appendOverlaps(const OverlapSet &from, AxisSymbol to);
  void resolveInto(AxisSymbol symbol, llvm::SmallVector<AxisSymbol> &result);
  OverlapSet overlappingOfRoot(AxisSymbol root);
};

} // namespace mlir::enzyme::axis

#endif // ENZYME_AD_JAX_DIALECT_AXIS_INCREMENTALAXISCONSTRAINTS_H
