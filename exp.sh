# Here are some example training and testing script.

#deeplabv3+
python train_backbone.py --dataset endovis2018 --arch puredeeplab18 --log_name DLV3PLUS_clean --t 1 --batch_size 4 --lr 1e-4 --gpu 0,3 --ver 0 --num_epochs 100 --data_type clean --h 256 --w 320
python train_backbone.py --dataset colon_oct --arch puredeeplab18 --log_name DLV3PLUS_oct --root_dir ./results/colon_oct --t 1 --batch_size 4 --lr 1e-4 --gpu 0,3 --ver 0 --num_epochs 100 --h 256 --w 256

python test.py --arch puredeeplab18 --log_name DLV3PLUS_clean_ver_0 --t 1 --gpu 0 --checkpoint 1 --h 256 --w 320
python test.py --dataset colon_oct --arch puredeeplab18 --log_name DLV3PLUS_oct_ver_0 --root_dir ./results/colon_oct --t 1 --gpu 0 --checkpoint 1 --h 256 --w 256

#STswin
python train_backbone.py --dataset endovis2018 --arch swinPlus --log_name STswin_clean --t 4 --batch_size 4 --lr 3e-5 --gpu 0,3 --ver 0 --num_epochs 100 --data_type clean --h 256 --w 320
python train_backbone.py --dataset colon_oct --arch swinPlus --log_name STswin_oct --root_dir ./results/colon_oct --t 4 --batch_size 4 --lr 3e-5 --gpu 0,3 --ver 0 --num_epochs 100 --h 256 --w 256

python test.py --arch swinPlus --log_name STswin_clean_ver_0 --t 4 --gpu 0 --checkpoint 1 --h 256 --w 320
python test.py --dataset colon_oct --arch swinPlus --log_name STswin_oct_ver_0 --root_dir ./results/colon_oct --t 4 --gpu 0 --checkpoint 1 --h 256 --w 256

#RAUNet
python train_backbone.py --dataset endovis2018 --arch RAUNet --log_name RAUNet_clean --t 1 --batch_size 4 --lr 1e-4 --gpu 0,3 --ver 0 --num_epochs 100 --data_type clean --h 256 --w 320
python train_backbone.py --dataset colon_oct --arch RAUNet --log_name RAUNet_oct --root_dir ./results/colon_oct --t 1 --batch_size 4 --lr 1e-4 --gpu 0,3 --ver 0 --num_epochs 100 --h 256 --w 256

python test.py --arch RAUNet --log_name RAUNet_clean_ver_0 --t 1 --gpu 0 --checkpoint 1 --h 256 --w 320
python test.py --dataset colon_oct --arch RAUNet --log_name RAUNet_oct_ver_0 --root_dir ./results/colon_oct --t 1 --gpu 0 --checkpoint 1 --h 256 --w 256

#MSS-TFAL
python train_MS_TFAL.py --dataset endovis2018 --arch puredeeplab18 --log_name MSS_TFAL_noisyver_0 --t 1 --batch_size 4 --lr 1e-4 --gpu 0,3 --ver 0 --iterations 56000 --data_type noisy --data_ver 0 --h 256 --w 320 --T1 4 --T2 6 --T3 8
python train_MS_TFAL.py --dataset colon_oct --arch puredeeplab18 --log_name MSS_TFAL_oct --root_dir ./results/colon_oct --t 1 --batch_size 4 --lr 1e-4 --gpu 0,3 --ver 2 --iterations 64000  --h 256 --w 256 --T1 2 --T2 3 --T3 4

python test.py --arch puredeeplab18 --log_name MS_TFAL_noisyver_3_ver_0 --t 1 --gpu 0 --checkpoint 1 --h 256 --w 320
python test.py --dataset colon_oct --arch puredeeplab18 --log_name MS_TFAL_oct_ver_0 --root_dir ./results/colon_oct --t 1 --gpu 0 --checkpoint 1 --h 256 --w 256
