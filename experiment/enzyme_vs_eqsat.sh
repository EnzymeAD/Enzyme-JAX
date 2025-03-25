#!/bin/bash

# Define the combinations of environment variables
configs=(
  # "export ENZYME_RULES=true MULTI_RULES=true EQSAT_RULES=false"
  "export ENZYME_RULES=true MULTI_RULES=false EQSAT_RULES=false"
)
platforms=("cpu" "gpu")
models=("bert" "gemma" "gpt2" "jaxmd" "kan1" "kan2" "llama" "maxtext" "nasrnn" "resnet" "searchlesschess" )
datetime=$(date '+%Y-%m-%d_%H:%M:%S')
filename=enzyme_vs_eqsat_$datetime.txt
num_repeats=5

export STATS_FILENAME=stats_enzyme_vs_eqsat_$datetime.csv
touch $STATS_FILENAME
echo "experiment_name,eqsat_time,segments" > $STATS_FILENAME

echo "Eqsat vs Enzyme" > $filename
echo "--------------------------" >> $filename

for model in "${models[@]}"; do
  for platform in "${platforms[@]}"; do
    for config in "${configs[@]}"; do
      eval "$config"
      export KERAS_BACKEND="jax"
      export EQSAT_PLATFORM=$platform

      if [ "$platform" == "gpu" ]; then
        COMMAND="CUDA_VISIBLE_DEVICES=2 python test/${model}.py"
      else
        COMMAND="JAX_PLATFORMS=cpu python test/${model}.py"
      fi

      for repeat in $(seq 1 $num_repeats); do
        export EXPERIMENT_NAME="${model}_enzyme-ablation-${platform}_${datetime}_run${repeat}"
        echo "Running $EXPERIMENT_NAME (repeat $repeat/$num_repeats)..." | tee -a $filename

        START_TIME=$(date +%s)
        eval "$COMMAND" >> $filename 2>&1
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        echo "$EXPERIMENT_NAME: ${DURATION} seconds" >> $filename
        echo "--------------------------------" >> $filename
      done
    done
  done
done
