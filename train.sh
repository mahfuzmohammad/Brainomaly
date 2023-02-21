FOLDER=$1

if [[ $# -eq 0 ]]; then
  echo "Correct usage: bash train.sh <dataset_folder_name>"
  exit
fi

python main.py --mode train --dataset MedicalData --image_size 192 \
                --image_dir data/${FOLDER} \
                --sample_dir ${FOLDER}/samples --log_dir ${FOLDER}/logs \
                --model_save_dir ${FOLDER}/models --result_dir ${FOLDER}/result \
                --num_iters 400000 --batch_size 16
