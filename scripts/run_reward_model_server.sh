MODEl_NAME=path/to/model
EVAL_PARALLEL=true

echo "run vllm server parallel..."

cuda_list=(0 1 2 3 4 5 6 7)
port_list=(12321 12321 12322 12322 12323 12323 12324 12324)
parallel=2

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
    --task classify \
    --hf-overrides '{"architectures": ["Gemma2ForCausalLM"]}' \
    --override-pooler-config '{"softmax": false}' \
    --tensor-parallel-size $parallel \
    --trust-remote-code &

    echo "run server on cuda $cuda, port $port "

done
