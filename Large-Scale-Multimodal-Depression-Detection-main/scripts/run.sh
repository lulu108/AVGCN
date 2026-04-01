python mainkfold.py --train True --num_folds 10 --epochs 125 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion lt --dataset dvlog-dataset

python mainkfold.py --train True --num_folds 10 --epochs 125 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion it --dataset dvlog-dataset

python mainkfold.py --train True --num_folds 10 --epochs 125 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion ia --dataset dvlog-dataset


python mainkfold.py --train True --num_folds 10 --epochs 125 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion lt --dataset lmvd-dataset

python mainkfold.py --train True --num_folds 10 --epochs 125 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion it --dataset lmvd-dataset

python mainkfold.py --train True --num_folds 10 --epochs 125 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion ia --dataset lmvd-dataset


python main.py --train True --epochs 225 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion lt --dataset dvlog-dataset

python main.py --train True --epochs 225 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion lt --dataset lmvd-dataset


python mainkfold.py --train True --num_folds 10 --epochs 125 --batch_size 16 --learning_rate 1e-4 --model DepMamba --fusion no_fusion --dataset dvlog-dataset

python mainkfold.py --train True --num_folds 10 --epochs 125 --batch_size 16 --learning_rate 1e-4 --model DepMamba --fusion no_fusion --dataset lmvd-dataset


# Mutual Transformer
python main.py --train True --epochs 225 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion MT --dataset dvlog-dataset

python mainkfold.py --train True --num_folds 10 --epochs 250 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion MT --dataset dvlog-dataset


python main.py --train True --epochs 225 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion MT --dataset lmvd-dataset

python mainkfold.py --train True --num_folds 10 --epochs 250 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion MT --dataset lmvd-dataset

## NEW
python mainkfold.py --train True --num_folds 10 --epochs 250 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion lt --dataset dvlog-dataset


python mainkfold.py --train True --num_folds 10 --start_fold 0 --epochs 250 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion audio --dataset dvlog-dataset
python mainkfold.py --train True --num_folds 10 --start_fold 0 --epochs 250 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion video --dataset dvlog-dataset

python mainkfold.py --train True --num_folds 10 --start_fold 1 --epochs 150 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion it --dataset dvlog-dataset
python mainkfold.py --train True --num_folds 10 --start_fold 1 --epochs 150 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion ia --dataset dvlog-dataset


python mainkfold.py --train True --num_folds 10 --start_fold 0 --epochs 250 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion audio --dataset lmvd-dataset
python mainkfold.py --train True --num_folds 10 --start_fold 0 --epochs 250 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion video --dataset lmvd-dataset


## testing ...
python mainkfold.py --num_folds 10 --start_fold 0 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion lt --dataset dvlog-dataset
python mainkfold.py --num_folds 10 --start_fold 0 --batch_size 16 --learning_rate 1e-5 --model MultiModalDepDet --fusion lt --dataset lmvd-dataset


### ( thats mean = Train (D-Vlog), Test (LMVD) )

python mainkfold.py --train True --num_folds 10 --start_fold 5 --epochs 225 --batch_size 8 --learning_rate 1e-5 --model MultiModalDepDet --fusion lt --dataset lmvd-dataset --cross_infer True