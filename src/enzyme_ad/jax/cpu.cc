#include "xla/service/custom_call_target_registry.h"

struct CallInfo {
  void (*run)(const void **);
};

void forwarding_custom_call(void *out, const void **in, const CallInfo *opaque,
                            size_t opaque_len, void *status) {
  opaque->run(in);
}

extern "C" void RegisterEnzymeXLACPUHandler() {
  xla::CustomCallTargetRegistry::Global()->Register(
      "enzymexla_compile_cpu", (void *)&forwarding_custom_call, "Host");
}
