#include "ProductLattice.h"
#include "Strategy.h"

using namespace mlir::enzyme::pact;

FactEntry &PropertyColumn::at(Level level) { return entries[level]; }

const FactEntry *PropertyColumn::get(Level level) const {
  auto it = entries.find(level);
  return it != entries.end() ? &it->second : nullptr;
}

void PropertyColumn::transition(Level level, FactState newState,
                                llvm::StringRef why) {
  auto &entry = at(level);
  entry.state = newState;
  if (!why.empty())
    entry.reason = why.str();
  trace.push_back({level, newState});
}

FactState PropertyColumn::currentState() const {
  auto it = entries.find(meetLevel);
  if (it != entries.end())
    return it->second.state;
  return FactState::Unknown;
}

PropertyColumn *ProductLattice::findColumn(llvm::StringRef key) {
  for (auto &col : columns)
    if (col.propertyKey == key)
      return &col;
  return nullptr;
}

const PropertyColumn *ProductLattice::findColumn(llvm::StringRef key) const {
  for (const auto &col : columns)
    if (col.propertyKey == key)
      return &col;
  return nullptr;
}

PropertyColumn &ProductLattice::addColumn(llvm::StringRef key,
                                          scheme::PropertyKind kind,
                                          Level meetLevel) {
  columns.emplace_back(key, kind, meetLevel);
  return columns.back();
}

bool ProductLattice::allResolved() const {
  for (const auto &col : columns) {
    auto *entry = col.get(col.meetLevel);
    if (!entry || !entry->isResolved())
      return false;
  }
  return true;
}

int ProductLattice::countMismatches() const {
  int count = 0;
  for (const auto &col : columns)
    if (col.currentState() == FactState::Mismatch)
      ++count;
  return count;
}

llvm::SmallVector<PropertyColumn *> ProductLattice::unresolvedColumns() {
  llvm::SmallVector<PropertyColumn *> result;
  for (auto &col : columns) {
    auto *entry = col.get(col.meetLevel);
    if (!entry || !entry->isResolved())
      result.push_back(&col);
  }
  return result;
}

bool ProductLattice::hasBlocking() const {
  for (const auto &col : columns) {
    auto *entry = col.get(col.meetLevel);
    if (entry && entry->matchResult.isBlocking())
      return true;
  }
  return false;
}