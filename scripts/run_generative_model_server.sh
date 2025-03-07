MODEl_NAME=path/to/model
EVAL_PARALLEL=true

echo "run vllm server parallel..."

cuda_list=(0 1 2 3 4 5 6 7)
port_list=(11101 11102 11103 11104 11105 11106 11107 11108)
parallel=1

if [ "$EVAL_PARALLEL" == "true" ]; then
    length=${#cuda_list[@]}
else
    length=1
fi

for ((i=0; i<$length; i=i+$parallel)); do
    port=${port_list[$i]}
    cuda=""
    for ((j=0; j<$parallel; j++))
    do
        if [ $j -eq 0 ]; then
            cuda="${cuda_list[$i+j]}"
        else
            cuda="$cuda,${cuda_list[$i+j]}"
        fi
    done

    CUDA_VISIBLE_DEVICES=$cuda python -m vllm.entrypoints.openai.api_server \
    --port $port \
    --model $MODEl_NAME \
    --tensor-parallel-size $parallel \
    --trust-remote-code &
    # --max-model-len 4096 \
    # --mem-fraction-static 0.5 \
    # --tokenizer $MODEl_NAME \
    # --max-model-len 16384 &
    # --enforce-eager &
    # --gpu-memory-utilization 0.9 &

    echo "run server on cuda $cuda, port $port "

done
