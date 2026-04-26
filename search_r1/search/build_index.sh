
corpus_file="/home/fuchengniu/sylvia/Search-R1/corpus/medcase.jsonl"
save_dir="/home/fuchengniu/sylvia/Search-R1/corpus/medcpt"
retriever_name=${RETRIEVER_NAME:-medcpt} # use medcpt for MedCPT article embeddings
retriever_model=${RETRIEVER_MODEL:-ncbi/MedCPT-Article-Encoder} # for MedCPT indexing use ncbi/MedCPT-Article-Encoder

# e5
# intfloat/e5-base-v2
# export RETRIEVER_NAME=medcpt
# export RETRIEVER_MODEL=ncbi/MedCPT-Article-Encoder

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 32 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding
