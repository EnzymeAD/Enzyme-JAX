#include "xla/service/custom_call_target_registry.h"

struct CallInfo {
  void (*run)(const void **);
};

struct CallInfoWithError {
  char *(*run)(const void **);
};

void forwarding_custom_call(void *out, const void **in, const CallInfo *opaque,
                            size_t opaque_len, void *status) {
  opaque->run(in);
}

void forwarding_custom_call_with_error(void *out, const void **in,
                                       const CallInfoWithError *opaque,
                                       size_t opaque_len, void *status) {
  char *err = opaque->run(in);
  if (err) {
    XlaCustomCallStatusSetFailure(status, err, strlen(err));
  }
}

extern "C" void RegisterEnzymeXLACPUHandler() {
  xla::CustomCallTargetRegistry::Global()->Register(
      "enzymexla_compile_cpu", (void *)&forwarding_custom_call, "Host");
  xla::CustomCallTargetRegistry::Global()->Register(
      "enzymexla_compile_cpu_with_error",
      (void *)&forwarding_custom_call_with_error, "Host");
}
