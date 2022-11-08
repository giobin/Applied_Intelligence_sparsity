cd /media/hdd/bonetta/projects/TransformerHuggingGio
source /home/bonetta/miniconda3/etc/profile.d/conda.sh
conda activate sparsity_venv

python3 main_trans.py --train_path=data/tsk1/train --validation_path=data/tsk1/dev --test_path=data/tsk1/test/ --type_dataset=tsk1 --log_interval=100 --eval_interval=300 -b 50 -e 40 --numworkers 8 -single_vocab --ckpt workdir/20-01-2022_17-56-34/pytorch_model.bin --configuration workdir/20-01-2022_17-56-34/config.json --path_vocab1 workdir/20-01-2022_17-56-34/L.vocab --work_dir l1_workdir --regularize --bleu_lower_bound 0.0603 --learning_rate 0.0001 --threshold 0.1 --reg_gamma 1e-05 --rtype l1
python3 main_trans.py --train_path=data/tsk1/train --validation_path=data/tsk1/dev --test_path=data/tsk1/test/ --type_dataset=tsk1 --log_interval=100 --eval_interval=300 -b 50 -e 40 --numworkers 8 -single_vocab --ckpt workdir/20-01-2022_17-56-34/pytorch_model.bin --configuration workdir/20-01-2022_17-56-34/config.json --path_vocab1 workdir/20-01-2022_17-56-34/L.vocab --work_dir l1_workdir --regularize --bleu_lower_bound 0.0603 --learning_rate 0.0005 --threshold 0.1 --reg_gamma 1e-05 --rtype l1
python3 main_trans.py --train_path=data/tsk1/train --validation_path=data/tsk1/dev --test_path=data/tsk1/test/ --type_dataset=tsk1 --log_interval=100 --eval_interval=300 -b 50 -e 40 --numworkers 8 -single_vocab --ckpt workdir/20-01-2022_17-56-34/pytorch_model.bin --configuration workdir/20-01-2022_17-56-34/config.json --path_vocab1 workdir/20-01-2022_17-56-34/L.vocab --work_dir l1_workdir --regularize --bleu_lower_bound 0.0603 --learning_rate 0.001 --threshold 0.1 --reg_gamma 1e-05 --rtype l1
echo "1 block finito"
python3 main_trans.py --train_path=data/tsk1/train --validation_path=data/tsk1/dev --test_path=data/tsk1/test/ --type_dataset=tsk1 --log_interval=100 --eval_interval=300 -b 50 -e 40 --numworkers 8 -single_vocab --ckpt workdir/20-01-2022_17-56-34/pytorch_model.bin --configuration workdir/20-01-2022_17-56-34/config.json --path_vocab1 workdir/20-01-2022_17-56-34/L.vocab --work_dir l1_workdir --regularize --bleu_lower_bound 0.0603 --learning_rate 0.0001 --threshold 0.2 --reg_gamma 1e-05 --rtype l1
python3 main_trans.py --train_path=data/tsk1/train --validation_path=data/tsk1/dev --test_path=data/tsk1/test/ --type_dataset=tsk1 --log_interval=100 --eval_interval=300 -b 50 -e 40 --numworkers 8 -single_vocab --ckpt workdir/20-01-2022_17-56-34/pytorch_model.bin --configuration workdir/20-01-2022_17-56-34/config.json --path_vocab1 workdir/20-01-2022_17-56-34/L.vocab --work_dir l1_workdir --regularize --bleu_lower_bound 0.0603 --learning_rate 0.0005 --threshold 0.2 --reg_gamma 1e-05 --rtype l1
python3 main_trans.py --train_path=data/tsk1/train --validation_path=data/tsk1/dev --test_path=data/tsk1/test/ --type_dataset=tsk1 --log_interval=100 --eval_interval=300 -b 50 -e 40 --numworkers 8 -single_vocab --ckpt workdir/20-01-2022_17-56-34/pytorch_model.bin --configuration workdir/20-01-2022_17-56-34/config.json --path_vocab1 workdir/20-01-2022_17-56-34/L.vocab --work_dir l1_workdir --regularize --bleu_lower_bound 0.0603 --learning_rate 0.001 --threshold 0.2 --reg_gamma 1e-05 --rtype l1
echo "2 block finito"
python3 main_trans.py --train_path=data/tsk1/train --validation_path=data/tsk1/dev --test_path=data/tsk1/test/ --type_dataset=tsk1 --log_interval=100 --eval_interval=300 -b 50 -e 40 --numworkers 8 -single_vocab --ckpt workdir/20-01-2022_17-56-34/pytorch_model.bin --configuration workdir/20-01-2022_17-56-34/config.json --path_vocab1 workdir/20-01-2022_17-56-34/L.vocab --work_dir l1_workdir --regularize --bleu_lower_bound 0.0603 --learning_rate 0.0001 --threshold 0.1 --reg_gamma 5e-06 --rtype l1
python3 main_trans.py --train_path=data/tsk1/train --validation_path=data/tsk1/dev --test_path=data/tsk1/test/ --type_dataset=tsk1 --log_interval=100 --eval_interval=300 -b 50 -e 40 --numworkers 8 -single_vocab --ckpt workdir/20-01-2022_17-56-34/pytorch_model.bin --configuration workdir/20-01-2022_17-56-34/config.json --path_vocab1 workdir/20-01-2022_17-56-34/L.vocab --work_dir l1_workdir --regularize --bleu_lower_bound 0.0603 --learning_rate 0.0005 --threshold 0.1 --reg_gamma 5e-06 --rtype l1
python3 main_trans.py --train_path=data/tsk1/train --validation_path=data/tsk1/dev --test_path=data/tsk1/test/ --type_dataset=tsk1 --log_interval=100 --eval_interval=300 -b 50 -e 40 --numworkers 8 -single_vocab --ckpt workdir/20-01-2022_17-56-34/pytorch_model.bin --configuration workdir/20-01-2022_17-56-34/config.json --path_vocab1 workdir/20-01-2022_17-56-34/L.vocab --work_dir l1_workdir --regularize --bleu_lower_bound 0.0603 --learning_rate 0.001 --threshold 0.1 --reg_gamma 5e-06 --rtype l1
echo "3 block finito"
python3 main_trans.py --train_path=data/tsk1/train --validation_path=data/tsk1/dev --test_path=data/tsk1/test/ --type_dataset=tsk1 --log_interval=100 --eval_interval=300 -b 50 -e 40 --numworkers 8 -single_vocab --ckpt workdir/20-01-2022_17-56-34/pytorch_model.bin --configuration workdir/20-01-2022_17-56-34/config.json --path_vocab1 workdir/20-01-2022_17-56-34/L.vocab --work_dir l1_workdir --regularize --bleu_lower_bound 0.0603 --learning_rate 0.0001 --threshold 0.2 --reg_gamma 5e-06 --rtype l1
python3 main_trans.py --train_path=data/tsk1/train --validation_path=data/tsk1/dev --test_path=data/tsk1/test/ --type_dataset=tsk1 --log_interval=100 --eval_interval=300 -b 50 -e 40 --numworkers 8 -single_vocab --ckpt workdir/20-01-2022_17-56-34/pytorch_model.bin --configuration workdir/20-01-2022_17-56-34/config.json --path_vocab1 workdir/20-01-2022_17-56-34/L.vocab --work_dir l1_workdir --regularize --bleu_lower_bound 0.0603 --learning_rate 0.0005 --threshold 0.2 --reg_gamma 5e-06 --rtype l1
python3 main_trans.py --train_path=data/tsk1/train --validation_path=data/tsk1/dev --test_path=data/tsk1/test/ --type_dataset=tsk1 --log_interval=100 --eval_interval=300 -b 50 -e 40 --numworkers 8 -single_vocab --ckpt workdir/20-01-2022_17-56-34/pytorch_model.bin --configuration workdir/20-01-2022_17-56-34/config.json --path_vocab1 workdir/20-01-2022_17-56-34/L.vocab --work_dir l1_workdir --regularize --bleu_lower_bound 0.0603 --learning_rate 0.001 --threshold 0.2 --reg_gamma 5e-06 --rtype l1
echo "finiti esperimenti!"

python3 main_trans.py --train_path=data/tsk1/train --validation_path=data/tsk1/dev --test_path=data/tsk1/test/ --type_dataset=tsk1 --log_interval=100 --eval_interval=300 -b 50 -e 4 --numworkers 8 -single_vocab --ckpt l2_workdir_allckpt/06-07-2022_09-50-00/25_299/pytorch_model.bin --configuration l2_workdir_allckpt/06-07-2022_09-50-00/25_299/config.json --path_vocab1 workdir/20-01-2022_17-56-34/L.vocab --work_dir l2_workdir_finetuning --learning_rate 0.001 --threshold 0.2 --reg_gamma 5e-06 --rtype l2 -p 5
python3 main_trans.py --train_path=data/tsk1/train --validation_path=data/tsk1/dev --test_path=data/tsk1/test/ --type_dataset=tsk1 --log_interval=100 --eval_interval=300 -b 50 -e 4 --numworkers 8 -single_vocab --ckpt l2_workdir_allckpt/06-07-2022_09-50-00/24_1199/pytorch_model.bin --configuration l2_workdir_allckpt/06-07-2022_09-50-00/24_1199/config.json --path_vocab1 workdir/20-01-2022_17-56-34/L.vocab --work_dir l2_workdir_finetuning --learning_rate 0.001 --threshold 0.2 --reg_gamma 5e-06 --rtype l2 -p 5
python3 main_trans.py --train_path=data/tsk1/train --validation_path=data/tsk1/dev --test_path=data/tsk1/test/ --type_dataset=tsk1 --log_interval=100 --eval_interval=300 -b 50 -e 4 --numworkers 8 -single_vocab --ckpt l2_workdir_allckpt/06-07-2022_09-50-00/18_599/pytorch_model.bin --configuration l2_workdir_allckpt/06-07-2022_09-50-00/18_599/config.json --path_vocab1 workdir/20-01-2022_17-56-34/L.vocab --work_dir l2_workdir_finetuning --learning_rate 0.001 --threshold 0.2 --reg_gamma 5e-06 --rtype l2 -p 5
python3 main_trans.py --train_path=data/tsk1/train --validation_path=data/tsk1/dev --test_path=data/tsk1/test/ --type_dataset=tsk1 --log_interval=100 --eval_interval=300 -b 50 -e 4 --numworkers 8 -single_vocab --ckpt l2_workdir_allckpt/06-07-2022_09-50-00/15_899/pytorch_model.bin --configuration l2_workdir_allckpt/06-07-2022_09-50-00/15_899/config.json --path_vocab1 workdir/20-01-2022_17-56-34/L.vocab --work_dir l2_workdir_finetuning --learning_rate 0.001 --threshold 0.2 --reg_gamma 5e-06 --rtype l2 -p 5
python3 main_trans.py --train_path=data/tsk1/train --validation_path=data/tsk1/dev --test_path=data/tsk1/test/ --type_dataset=tsk1 --log_interval=100 --eval_interval=300 -b 50 -e 4 --numworkers 8 -single_vocab --ckpt l2_workdir_allckpt/06-07-2022_09-50-00/14_1199/pytorch_model.bin --configuration l2_workdir_allckpt/06-07-2022_09-50-00/14_1199/config.json --path_vocab1 workdir/20-01-2022_17-56-34/L.vocab --work_dir l2_workdir_finetuning --learning_rate 0.001 --threshold 0.2 --reg_gamma 5e-06 --rtype l2 -p 5