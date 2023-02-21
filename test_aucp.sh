folder=$1
iter=$2

if [[ $# -lt 2 ]]; then
  echo "Correct usage: bash test_aucp.sh <dataset_folder_name> <model_iteration>"
  exit
fi

NEPTUNE_ID=$(cat ${folder}/neptune_id)
echo ${folder} ${iter} ${NEPTUNE_ID}

python main.py --mode testAUCp --dataset MedicalData --image_size 192 \
                --image_dir data/${folder} \
                --sample_dir ${folder}/samples --log_dir ${folder}/logs \
                --model_save_dir ${folder}/models --result_dir ${folder}/result \
                --test_iters ${iter} --batch_size 1 --num_workers 2 \
                --neptune_id ${NEPTUNE_ID} --neptune_key Test_AUCp
