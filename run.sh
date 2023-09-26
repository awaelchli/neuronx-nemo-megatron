cd neuronx-nemo-megatron/nemo
if [ "$LOCAL_RANK" = "0" ]; then
    # pip3 install ./build/*.whl
    pip install -e .
else
    sleep 20
fi


# TODO: why is this needed?
# pip install -U boto botocore

cd examples/nlp/language_modeling
export HOME=/home/zeus/content
bash gpt_test.sh
