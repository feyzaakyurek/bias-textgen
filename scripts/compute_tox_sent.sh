#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-5
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=bias-toxsent


PTH="/home/gridsan/akyurek/git/bias-textgen"

cnt=0
for MAXLEN in 10 20 30 40 50; do
for NUMGEN in 20 ; do
for PROMPTSET in "honest"; do
for DOMAIN in "gender"; do
for TEMP in 1.0; do
# for CONSIDER in all worst1 worst5; do
(( cnt++ ))
if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then

    EXPFOLDER="$PTH/outputs/$PROMPTSET/$DOMAIN"
    mkdir -p $EXPFOLDER
    EXPNAME="len_${MAXLEN}_num_${NUMGEN}_temp_${TEMP}" # _temp_${TEMP} fix temp

    python compute_tox_sent.py --test_file $EXPFOLDER/${EXPNAME}_gens.csv \
                               --save_path $EXPFOLDER \
                               --prompt_set $PROMPTSET \
                               --prompt_domain $DOMAIN \
                               --category ${EXPNAME} > $EXPFOLDER/${EXPNAME}_sent_tox.log 2>&1
fi
# done
done
done
done
done
done

# MAXLEN=50
# NUMGEN=20
# TEMP=1
# PROMPTSET=bold
# DOMAIN=religious_ideology

# EXPFOLDER="$PTH/outputs/$PROMPTSET/$DOMAIN"
# EXPNAME="len_${MAXLEN}_num_${NUMGEN}"

# python compute_tox_sent.py --input_file $EXPFOLDER/${EXPNAME}_gens.csv \
#                                --save_path $EXPFOLDER \
#                                --prompt_set $PROMPTSET \
#                                --prompt_domain $DOMAIN \
#                                --category $EXPNAME