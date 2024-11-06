#!/bin/bash

thresholds=(100 500 1000 5000)
platforms=("cpu" "gpu")
models=("llama" "maxtext" "jaxmd")

echo "Segmentation" > segmentation.txt
echo "--------------------------" >> segmentation.txt

for model in "${models[@]}"; do
  for platform in "${platforms[@]}"; do
    for threshold in "${thresholds[@]}"; do
      export EXPERIMENT_NAME="${model}_tau=${threshold}-${platform}"
      export KERAS_BACKEND="jax"
      export SEGMENTATION_THRESHOLD=$threshold
      export EQSAT_PLATFORM=$platform

      if [ "$platform" == "gpu" ]; then
        COMMAND="CUDA_VISIBLE_DEVICES=2 python test/${model}.py"
      else
        COMMAND="JAX_PLATFORMS=cpu python test/${model}.py"
      fi

      # time the command and redirect output to segmentation.txt
      echo "Running $EXPERIMENT_NAME..." | tee -a segmentation.txt
      START_TIME=$(date +%s)
      eval "$COMMAND"
      END_TIME=$(date +%s)
      DURATION=$((END_TIME - START_TIME))

      # append the timing to segmentation.txt
      echo "$EXPERIMENT_NAME: ${DURATION} seconds" >> segmentation.txt
      echo "--------------------------------" >> segmentation.txt
    done
  done
done
