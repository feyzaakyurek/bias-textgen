#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-75
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=bias-textgen


PTH="/home/gridsan/akyurek/git/bias-textgen"

cnt=0
for MAXLEN in 10 20 30 40 50; do
for NUMGEN in 20 ; do
for PROMPTSET in "bold"; do
for DOMAIN in "race" "gender" "religious_ideology"; do
for TEMP in 0.2 0.4 0.6 0.8; do
# for CONSIDER in all worst1 worst5; do
(( cnt++ ))
if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then

    EXPFOLDER="$PTH/outputs/$PROMPTSET/$DOMAIN"
    mkdir -p $EXPFOLDER
    EXPNAME="len_${MAXLEN}_num_${NUMGEN}_temp_${TEMP}"

    python complete_prompts.py --model_name gpt2 \
                               --model_path "$PTH/gpt2" \
                               --save_path $EXPFOLDER \
                               --prompt_set $PROMPTSET \
                               --prompt_domain $DOMAIN \
                               --temperature $TEMP \
                               --max_length $MAXLEN \
                               --num_gens $NUMGEN > $EXPFOLDER/$EXPNAME.log 2>&1
fi
# done
done
done
done
done
done

# MAXLEN=50
# NUMGEN=1
# PROMPTSET=bold
# DOMAIN=religious_ideology
# EXPFOLDER="$PTH/outputs/$PROMPTSET/$DOMAIN"
# EXPNAME="len_${MAXLEN}_num_${NUMGEN}"
# python complete_prompts.py --model_name gpt2 \
#     --model_path "$PTH/gpt2" \
#     --save_path $EXPFOLDER \
#     --prompt_set $PROMPTSET \
#     --prompt_domain $DOMAIN \
#     --num_gens $NUMGEN # > $EXPFOLDER/$EXPNAME.log 2>&1