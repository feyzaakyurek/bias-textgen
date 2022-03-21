#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-4
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=bias-regard


PTH="/home/gridsan/akyurek/git/bias-textgen"

cnt=0
for MAXLEN in 10 30 40 50; do
for NUMGEN in 20 ; do
for PROMPTSET in "bold"; do
for DOMAIN in "gender"; do
# for TEMP in 0.2 0.4 0.6 0.8; do
(( cnt++ ))
if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then

    EXPFOLDER="$PTH/outputs/$PROMPTSET/$DOMAIN"
    mkdir -p $EXPFOLDER
    EXPNAME="len_${MAXLEN}_num_${NUMGEN}_temp_1" # fix temp

    python compute_tox_sent.py --test_file $EXPFOLDER/${EXPNAME}_sent_tox.csv \
                               --save_path $EXPFOLDER \
                               --prompt_set $PROMPTSET \
                               --prompt_domain $DOMAIN \
                               --category ${EXPNAME} \
                               --regard \
                               --data_dir $PTH/nlgbias/data/regard \
                               --model_type bert \
                               --model_name_or_path $PTH/nlgbias/models/bert_regard_v2_large/checkpoint-300 \
                               --output_dir $PTH/nlgbias/models/bert_regard_v2_large \
                               --max_seq_length 128 \
                               --do_predict \
                               --do_lower_case \
                               --per_gpu_eval_batch_size 32 \
                               --model_version 2 > $EXPFOLDER/${EXPNAME}_regard.log 2>&1
fi
# done
done
done
done
done

# MAXLEN=20
# NUMGEN=20
# TEMP=1
# PROMPTSET=bold
# DOMAIN=gender

# EXPFOLDER="$PTH/outputs/$PROMPTSET/$DOMAIN"
# EXPNAME="len_${MAXLEN}_num_${NUMGEN}_temp_1"

# python compute_tox_sent.py --test_file $EXPFOLDER/${EXPNAME}_sent_tox.csv \
#                                --save_path $EXPFOLDER \
#                                --prompt_set $PROMPTSET \
#                                --prompt_domain $DOMAIN \
#                                --category ${EXPNAME} \
#                                --regard \
#                                --data_dir $PTH/nlgbias/data/regard \
#                                --model_type bert \
#                                --model_name_or_path $PTH/nlgbias/models/bert_regard_v2_large/checkpoint-300 \
#                                --output_dir $PTH/nlgbias/models/bert_regard_v2_large \
#                                --max_seq_length 128 \
#                                --do_predict \
#                                --do_lower_case \
#                                --per_gpu_eval_batch_size 32 \
#                                --model_version 2