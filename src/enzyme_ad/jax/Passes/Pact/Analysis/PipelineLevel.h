#ifndef PACT_ANALYSIS_PIPELINE_LEVEL_H
#define PACT_ANALYSIS_PIPELINE_LEVEL_H

#include "llvm/ADT/StringRef.h"

namespace mlir::enzyme::pact {

enum class Level : int {
  L0 = 0, // LLVM IR
  L1 = 1, // target dialect
  L2 = 2, // Annotated Semantic
  L3 = 3, // Semantic Dialect
  L4 = 4, // GPU Dialect
  L5 = 5  // LLVM Dialect
};

constexpr int kNumLevels = 6;

inline bool operator>=(Level a, Level b) {
  return static_cast<int>(a) >= static_cast<int>(b);
}
inline bool operator>(Level a, Level b) {
  return static_cast<int>(a) > static_cast<int>(b);
}
inline bool operator<=(Level a, Level b) {
  return static_cast<int>(a) <= static_cast<int>(b);
}
inline bool operator<(Level a, Level b) {
  return static_cast<int>(a) < static_cast<int>(b);
}

inline llvm::StringRef levelName(Level l) {
  switch (l) {
  case Level::L0:
    return "L0";
  case Level::L1:
    return "L1";
  case Level::L2:
    return "L2";
  case Level::L3:
    return "L3";
  case Level::L4:
    return "L4";
  case Level::L5:
    return "L5";
  }
  return "L?";
}

} // namespace mlir::enzyme::pact

#endif // PACT_ANALYSIS_PIPELINE_LEVEL_H