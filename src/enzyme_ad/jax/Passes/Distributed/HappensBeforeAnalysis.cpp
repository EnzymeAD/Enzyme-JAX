#include "src/enzyme_ad/jax/Passes/Distributed/HappensBeforeAnalysis.h"

#include <optional>
#include <queue>

namespace mlir {
namespace enzyme {
namespace distributed {

HappensBeforeAnalysis::HappensBeforeAnalysis(Operation *op) {
  // Require the op to implement the LaneOp interface
  LaneOpInterface laneOp = dyn_cast<LaneOpInterface>(op);
  assert(laneOp && "HappensBeforeAnalysis requires an operation that "
                   "implements LaneOpInterface");

  // Step 1: Scan through all lanes and merge any simultaneous operations.
  // By doing this first, we don't need to worry about unioning
  // happens-before edges when merging.
  scanSimultaneousOperations(laneOp);

  // Step 2: For each lane add any happens-before edges from lane ops
  // and from the ordering of operations within the lane. During this
  // stage we can also detect any cycles within an equivalence class,
  // indicating a malformed happens-before.
  scanHappensBeforeEdges(laneOp);

  // Compute topological order; also detects cycles/deadlocks.
  computeTopologicalOrder();
}

void HappensBeforeAnalysis::markSimultaneous(Operation *a, Operation *b) {
  simultaneousClasses.unionSets(a, b);
}

void push_if_not_present(llvm::SmallVector<Operation *> &vec, Operation *op) {
  if (!llvm::is_contained(vec, op)) {
    vec.push_back(op);
  }
}

void HappensBeforeAnalysis::addHappensBeforeEdge(Operation *a, Operation *b) {
  a = simultaneousClasses.getOrInsertLeaderValue(a);
  b = simultaneousClasses.getOrInsertLeaderValue(b);
  if (a == b) {
    // Already in the same simultaneous class, an edge is illegal!
    a->emitError()
        << "Illegal happens-before edge between simultaneous operations"
        << "with other operation" << b;
    assert(false &&
           "Illegal happens-before edge between simultaneous operations");
    return;
  }
  push_if_not_present(predecessorsMap[b], a);
  push_if_not_present(successorsMap[a], b);
}

/**
 * Used for buliding the simultaneous class.
 */
llvm::SmallVector<EventOpInterface> all_events_simultaneous_with(Value handle) {
  llvm::SmallVector<EventOpInterface> result;
  for (Operation *user : handle.getUsers()) {
    if (auto eventOp = dyn_cast<EventOpInterface>(user)) {
      if (llvm::is_contained(eventOp.simultaneousWith(), handle)) {
        result.push_back(eventOp);
      }
    }
  }
  return result;
}
/**
 * Used for querying the root of the simultaneous class, if any.
 */
std::optional<EventOpInterface> any_event_simultaneous_with(Value handle) {
  for (Operation *user : handle.getUsers()) {
    if (auto eventOp = dyn_cast<EventOpInterface>(user)) {
      if (llvm::is_contained(eventOp.simultaneousWith(), handle)) {
        return eventOp;
      }
    }
  }
  return std::nullopt;
}

void HappensBeforeAnalysis::scanSimultaneousOperations(LaneOpInterface laneOp) {
  // Strategy:
  // look at every op in every lane for their simultatnous handles.
  // Look at every user of the handle that is an EventOpInterface.
  // Check if those users are simultanous with the handle too.
  // Union any simultaneous operations together.
  for (Region *lane : laneOp.getLanes()) {
    for (Operation &op : lane->getOps()) {
      // If it implements the Event op interface we can query for the
      // handles it is simultaenous with.
      if (auto eventOp = dyn_cast<EventOpInterface>(op)) {
        for (Value handle : eventOp.simultaneousWith()) {
          for (EventOpInterface otherEventOp :
               all_events_simultaneous_with(handle)) {
            markSimultaneous(&op, otherEventOp.getOperation());
          }
        }
      }
    }
  }
}

void HappensBeforeAnalysis::scanHappensBeforeEdges(LaneOpInterface laneOp) {
  // Strategy: more complex!
  // For each lane, for now assert a single basic block / linear control
  // flow. (TODO: check or handle nested control flow) Add a happens-before edge
  // between each op and its successor in the the block, excepting successors
  // that belong to the same simultaneous class. We assert that all members of
  // the same simultaneous class are contiguous and all concurrent with each
  // other. Otherwise, as we are scanning, add any happens-before edges from
  // events to any simulatenous events.
  for (Region *lane : laneOp.getLanes()) {
    assert(llvm::hasSingleElement(*lane) &&
           "HappensBeforeAnalysis currently requires single-block lanes");
    Block &block = lane->front();
    if (block.empty())
      continue;
    // at least one op, can safely take start + 1;
    llvm::SmallVector<Operation *> lastOpsInSimultaneousClass;
    Operation *current_root;
    // Prep using the first op.
    current_root = simultaneousClasses.getOrInsertLeaderValue(&block.front());
    lastOpsInSimultaneousClass.push_back(&block.front());
    for (Operation &op : llvm::drop_begin(block)) {
      Operation *next_root = simultaneousClasses.getOrInsertLeaderValue(&op);
      if (next_root != current_root) {
        // New simultaneous class, add happens-before edges from all ops in the
        // previous simultaneous class to this new simultaneous class.
        for (Operation *lastOp : lastOpsInSimultaneousClass) {
          addHappensBeforeEdge(lastOp, &op);
        }
        lastOpsInSimultaneousClass.clear();
        current_root = next_root;
      } else {
        // Same simultaenous class. Require that this op is
        // concurrent with all previous ops.
        EventOpInterface eventOp = cast<EventOpInterface>(&op);
        bool concurrent = true;
        for (Operation *lastOp : lastOpsInSimultaneousClass) {
          if (!eventOp.concurrentWith(lastOp)) {
            concurrent = false;
            break;
          }
        }
        if (!concurrent) {
          eventOp->emitOpError()
              << "Op is not concurrent with all previous ops "
                 "in the same simultaneous class and lane";
          assert(false &&
                 "All simultanous ops in the same lane must be concurrent");
        }
      }
      // Add this op to the current simultaneous class.
      lastOpsInSimultaneousClass.push_back(&op);

      // Normal explicit happens-before edges if this op is an event:
      if (auto eventOp = dyn_cast<EventOpInterface>(&op)) {
        for (Value handle : eventOp.happensAfter()) {
          auto representitive_other = any_event_simultaneous_with(handle);
          if (representitive_other) {
            addHappensBeforeEdge(representitive_other->getOperation(),
                                 &op);
          }
        }
      }
    }
  }
}

bool HappensBeforeAnalysis::happensBefore(Operation *a, Operation *b) const {
  if (!a || !b)
    return false;
  if (!simultaneousClasses.contains(a) || !simultaneousClasses.contains(b))
    return false;

  a = simultaneousClasses.getLeaderValue(a);
  b = simultaneousClasses.getLeaderValue(b);
  if (a == b)
    return false;

  auto succIt = successorsMap.find(a);
  if (succIt == successorsMap.end())
    return false;
  return llvm::is_contained(succIt->second, b);
}

bool HappensBeforeAnalysis::simultaneousWith(Operation *a, Operation *b) const {
  if (!a || !b)
    return false;
  if (!simultaneousClasses.contains(a) || !simultaneousClasses.contains(b))
    return false;
  return simultaneousClasses.getLeaderValue(a) ==
         simultaneousClasses.getLeaderValue(b);
}

llvm::SmallVector<Operation *>
HappensBeforeAnalysis::classList(Operation *classRoot) const {
  llvm::SmallVector<Operation *> result;
  if (!classRoot || !simultaneousClasses.contains(classRoot))
    return result;

  classRoot = simultaneousClasses.getLeaderValue(classRoot);
  for (Operation *member : simultaneousClasses.members(classRoot))
    result.push_back(member);
  return result;
}

Operation *HappensBeforeAnalysis::classRoot(Operation *classMember) const {
  if (!classMember || !simultaneousClasses.contains(classMember))
    return nullptr;
  return simultaneousClasses.getLeaderValue(classMember);
}

llvm::SmallVector<Operation *>
HappensBeforeAnalysis::predecessorClasses(Operation *a) const {
  if (!a || !simultaneousClasses.contains(a))
    return {};
  a = simultaneousClasses.getLeaderValue(a);
  auto predIt = predecessorsMap.find(a);
  if (predIt == predecessorsMap.end())
    return {};
  return predIt->second;
}

llvm::SmallVector<Operation *>
HappensBeforeAnalysis::successorClasses(Operation *a) const {
  if (!a || !simultaneousClasses.contains(a))
    return {};
  a = simultaneousClasses.getLeaderValue(a);
  auto succIt = successorsMap.find(a);
  if (succIt == successorsMap.end())
    return {};
  return succIt->second;
}

void HappensBeforeAnalysis::computeTopologicalOrder() {
  // Collect all class roots (leaders in the equivalence classes).
  llvm::SmallVector<Operation *> allRoots;
  for (const auto *ecVal : simultaneousClasses)
    if (ecVal->isLeader())
      allRoots.push_back(ecVal->getData());

  // Kahn's algorithm: process nodes whose predecessors are all done.
  llvm::DenseMap<Operation *, int64_t> indegree;
  std::queue<Operation *> ready;
  for (Operation *root : allRoots) {
    auto predIt = predecessorsMap.find(root);
    int64_t din =
        (predIt != predecessorsMap.end()) ? (int64_t)predIt->second.size() : 0;
    indegree[root] = din;
    if (din == 0)
      ready.push(root);
  }

  topologicalOrder.clear();
  while (!ready.empty()) {
    Operation *root = ready.front();
    ready.pop();
    topologicalOrder.push_back(root);
    auto succIt = successorsMap.find(root);
    if (succIt == successorsMap.end())
      continue;
    for (Operation *succ : succIt->second)
      if (--indegree[succ] == 0)
        ready.push(succ);
  }

  // If not all roots were processed, there is a cycle — deadlock.
  assert(topologicalOrder.size() == allRoots.size() &&
         "HappensBeforeAnalysis detected cycle/deadlock in happens-before "
         "graph");
}

const llvm::SmallVector<Operation *> &
HappensBeforeAnalysis::classesInTopologicalOrder() const {
  return topologicalOrder;
}

bool HappensBeforeAnalysis::isInvalidated(
    const AnalysisManager::PreservedAnalyses &pa) {
  // The analysis is invalidated if it's not explicitly preserved
  return !pa.isPreserved<HappensBeforeAnalysis>();
}

} // namespace distributed
} // namespace enzyme
} // namespace mlir
