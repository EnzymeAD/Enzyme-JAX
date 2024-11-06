#!/bin/bash

platforms=("cpu" "gpu")
models=("llama" "maxtext" "jaxmd")

echo "Basline" > baseline.txt
echo "--------------------------" >> baseline.txt

for model in "${models[@]}"; do
  for platform in "${platforms[@]}"; do
    export EXPERIMENT_NAME="${model}-${platform}"
    export KERAS_BACKEND="jax"
    export EQSAT_PLATFORM=$platform

    if [ "$platform" == "gpu" ]; then
      COMMAND="CUDA_VISIBLE_DEVICES=2 python test/${model}.py"
    else
      COMMAND="JAX_PLATFORMS=cpu python test/${model}.py"
    fi

    # time the command and redirect output to baseline.txt
    echo "Running $EXPERIMENT_NAME..." | tee -a baseline.txt
    START_TIME=$(date +%s)
    eval "$COMMAND"
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    # append the timing to baseline.txt
    echo "$EXPERIMENT_NAME: ${DURATION} seconds" >> baseline.txt
    echo "--------------------------------" >> baseline.txt
  done
done
