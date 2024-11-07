#!/bin/bash

# Define the combinations of environment variables
configs=(
  "ENZYME_RULES=true MULTI_RULES=true EQSAT_RULES=false"
  "ENZYME_RULES=true MULTI_RULES=false EQSAT_RULES=false"
)
platforms=("cpu" "gpu")
models=("llama" "maxtext" "jaxmd")

echo "Eqsat vs Enzyme" > eqsat_vs_enzyme.txt
echo "--------------------------" >> eqsat_vs_enzyme.txt

for model in "${models[@]}"; do
  for platform in "${platforms[@]}"; do
    for config in "${configs[@]}"; do
      # Set environment variables for each configuration
      eval "$config"
      export EXPERIMENT_NAME="${model}_${config// /_}-${platform}"
      export KERAS_BACKEND="jax"
      export EQSAT_PLATFORM=$platform

      if [ "$platform" == "gpu" ]; then
        COMMAND="CUDA_VISIBLE_DEVICES=2 python test/${model}.py"
      else
        COMMAND="JAX_PLATFORMS=cpu python test/${model}.py"
      fi

      # Time the command and redirect output to eqsat_vs_enzyme.txt
      echo "Running $EXPERIMENT_NAME..." | tee -a eqsat_vs_enzyme.txt
      START_TIME=$(date +%s)
      eval "$COMMAND" >> eqsat_vs_enzyme.txt 2>&1
      END_TIME=$(date +%s)
      DURATION=$((END_TIME - START_TIME))

      # Append the timing to eqsat_vs_enzyme.txt
      echo "$EXPERIMENT_NAME: ${DURATION} seconds" >> eqsat_vs_enzyme.txt
      echo "--------------------------------" >> eqsat_vs_enzyme.txt
    done
  done
done
