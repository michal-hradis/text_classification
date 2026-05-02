SCRIPT_DIR=/home/ihradis/projects/2026-05-01_text_classification/text_classification
CONFIG_DIR=$SCRIPT_DIR/configs
export HF_HOME=/home/ihradis/projects/2026-05-01_text_classification/huggingface_cache
export HF_HUB_TOKEN=


for pooling in cls cls mean
do
for model in HPLT/hplt_bert_base_2_0_ces-Latn HPLT/hplt_bert_base_cs jhu-clsp/mmBERT-small jhu-clsp/mmBERT-base ufal/robeczech-base \
    UWB-AIR/Czert-B-base-cased UWB-AIR/Czert-A-base-uncased Seznam/small-e-czech FacebookAI/xlm-roberta-base mpolacek/ElectraCzech-small \
    UWB-AIR/barticzech-1.0 HPLT/hplt_gpt_bert_base_3_0_ces_Latn FacebookAI/xlm-roberta-large \
    Seznam/simcse-dist-mpnet-czeng-cs-en Seznam/dist-mpnet-paracrawl-cs-en Seznam/dist-mpnet-czeng-cs-en \
    Seznam/simcse-dist-mpnet-paracrawl-cs-en Seznam/simcse-retromae-small-cs Seznam/simcse-small-e-czech
do
    MODEL_ONLY=$(echo $model | cut -d "/" -f 2)
    if [[ -f "$MODEL_ONLY.yaml" ]]; then
        python ${SCRIPT_DIR}/train.py ${CONFIG_DIR}/$MODEL_ONLY.yaml clearml.enabled=true clearml.project="text-classification" clearml.task_name=${pooling}-$MODEL_ONLY model.pooling=$pooling
    else
        python ${SCRIPT_DIR}/train.py ${CONFIG_DIR}/base.yaml model.name_or_path=$model clearml.enabled=true clearml.project="text-classification" clearml.task_name=${pooling}-$MODEL_ONLY model.pooling=$pooling
    fi
done
done
