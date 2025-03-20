#!/bin/bash

# Define the combinations of environment variables
configs=(
  # "export ENZYME_RULES=true MULTI_RULES=true EQSAT_RULES=false"
  "export ENZYME_RULES=true MULTI_RULES=false EQSAT_RULES=false"
)
platforms=("cpu" "gpu")
models=("llama" "maxtext" "jaxmd")
datetime=$(date '+%Y-%m-%d_%H:%M:%S')
filename=enzyme_vs_eqsat_$datetime.txt

export STATS_FILENAME=stats_enzyme_vs_eqsat_$datetime.csv
touch $STATS_FILENAME
echo "experiment_name,eqsat_time,segments" > $STATS_FILENAME

echo "Eqsat vs Enzyme" > $filename
echo "--------------------------" >> $filename

for model in "${models[@]}"; do
  for platform in "${platforms[@]}"; do
    for config in "${configs[@]}"; do
      # Set environment variables for each configuration
      eval "$config"
      export EXPERIMENT_NAME="${model}_enzyme-ablation-${platform}_$datetime"
      export KERAS_BACKEND="jax"
      export EQSAT_PLATFORM=$platform

      if [ "$platform" == "gpu" ]; then
        COMMAND="CUDA_VISIBLE_DEVICES=2 python test/${model}.py"
      else
        COMMAND="JAX_PLATFORMS=cpu python test/${model}.py"
      fi

      # Time the command and redirect output to eqsat_vs_enzyme.txt
      echo "Running $EXPERIMENT_NAME..." | tee -a $filename
      START_TIME=$(date +%s)
      eval "$COMMAND" >> $filename 2>&1
      END_TIME=$(date +%s)
      DURATION=$((END_TIME - START_TIME))

      # Append the timing to eqsat_vs_enzyme.txt
      echo "$EXPERIMENT_NAME: ${DURATION} seconds" >> $filename
      echo "--------------------------------" >> $filename
    done
  done
done
