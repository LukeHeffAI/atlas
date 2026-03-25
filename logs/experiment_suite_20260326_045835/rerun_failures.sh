#!/bin/bash
# Auto-generated script to rerun failed experiments
# Generated: Thu Mar 26 05:04:48 AM ACDT 2026
set -uo pipefail

echo "Rerunning: [clip] ViT-B-32: Fine-tune on all datasets"
python src/finetune.py --model=ViT-B-32 --clip-backend=clip --checkpoint-root=checkpoints_clip

echo "Rerunning: [clip] ViT-B-32: Learn task negation"
python src/learn_task_negation.py --model=ViT-B-32 --clip-backend=clip --checkpoint-root=checkpoints_clip --blockwise-coef

echo "Rerunning: [clip] ViT-B-32: Learn task addition"
python src/learn_task_addition.py --model=ViT-B-32 --clip-backend=clip --checkpoint-root=checkpoints_clip --blockwise-coef

echo "Rerunning: [clip] ViT-B-32: Test-time adaptation (UFM)"
python src/learn_ufm.py --model=ViT-B-32 --clip-backend=clip --checkpoint-root=checkpoints_clip --logdir=results_clip/ --blockwise-coef

echo "Rerunning: [clip] ViT-B-32: Few-shot adaptation (2 shots)"
python src/learn_few_shots.py --model=ViT-B-32 --clip-backend=clip --checkpoint-root=checkpoints_clip --logdir=results_clip/ --blockwise-coef --subsample 2

echo "Rerunning: [clip] ViT-B-32: Few-shot + tip adapter (2 shots)"
python src/learn_few_shots.py --model=ViT-B-32 --clip-backend=clip --checkpoint-root=checkpoints_clip --logdir=results_clip/ --blockwise-coef --subsample 2 --adapter tip

echo "Rerunning: [clip] ViT-B-32: Few-shot + lpp adapter (2 shots)"
python src/learn_few_shots.py --model=ViT-B-32 --clip-backend=clip --checkpoint-root=checkpoints_clip --logdir=results_clip/ --blockwise-coef --subsample 2 --adapter lpp

echo "Rerunning: [clip] ViT-B-32: aTLAS x K (partition=10, subsample=0.25)"
python src/learn_few_shots.py --model=ViT-B-32 --clip-backend=clip --checkpoint-root=checkpoints_clip --logdir=results_clip/ --partition 10 --subsample 0.25

echo "Rerunning: [clip] ViT-B-32: Meta-train text-to-coefficient hypernetwork"
python src/learn_text_to_coef.py --model=ViT-B-32 --clip-backend=clip --save=checkpoints_clip/ViT-B-32 --hypernetwork-arch=small --text-source=manual --meta-train-datasets=CIFAR10,EuroSAT,DTD,GTSRB,SVHN,Food101 --meta-val-datasets=Caltech101,Flowers102 --meta-epochs=3 --episodes-per-epoch=3 --blockwise-coef

echo "Rerunning: [clip] ViT-B-32: Meta-train multi-modal hypernetwork"
python src/learn_multimodal_to_coef.py --model=ViT-B-32 --clip-backend=clip --save=checkpoints_clip/ViT-B-32 --hypernetwork-arch=small --fusion-mode=concat --num-shots=4 --image-pooling=mean --text-input-mode=dataset --variable-shots --meta-train-datasets=CIFAR10,EuroSAT,DTD,GTSRB,SVHN,Food101 --meta-val-datasets=Caltech101,Flowers102 --meta-epochs=3 --episodes-per-epoch=3 --blockwise-coef

echo "Rerunning: [openclip] ViT-B-32: Fine-tune on all datasets"
python src/finetune.py --model=ViT-B-32 --clip-backend=openclip --checkpoint-root=checkpoints_openclip

echo "Rerunning: [openclip] ViT-B-32: Learn task negation"
python src/learn_task_negation.py --model=ViT-B-32 --clip-backend=openclip --checkpoint-root=checkpoints_openclip --blockwise-coef

echo "Rerunning: [openclip] ViT-B-32: Learn task addition"
python src/learn_task_addition.py --model=ViT-B-32 --clip-backend=openclip --checkpoint-root=checkpoints_openclip --blockwise-coef

echo "Rerunning: [openclip] ViT-B-32: Test-time adaptation (UFM)"
python src/learn_ufm.py --model=ViT-B-32 --clip-backend=openclip --checkpoint-root=checkpoints_openclip --logdir=results_openclip/ --blockwise-coef

echo "Rerunning: [openclip] ViT-B-32: Few-shot adaptation (2 shots)"
python src/learn_few_shots.py --model=ViT-B-32 --clip-backend=openclip --checkpoint-root=checkpoints_openclip --logdir=results_openclip/ --blockwise-coef --subsample 2

echo "Rerunning: [openclip] ViT-B-32: Few-shot + tip adapter (2 shots)"
python src/learn_few_shots.py --model=ViT-B-32 --clip-backend=openclip --checkpoint-root=checkpoints_openclip --logdir=results_openclip/ --blockwise-coef --subsample 2 --adapter tip

echo "Rerunning: [openclip] ViT-B-32: Few-shot + lpp adapter (2 shots)"
python src/learn_few_shots.py --model=ViT-B-32 --clip-backend=openclip --checkpoint-root=checkpoints_openclip --logdir=results_openclip/ --blockwise-coef --subsample 2 --adapter lpp

echo "Rerunning: [openclip] ViT-B-32: aTLAS x K (partition=10, subsample=0.25)"
python src/learn_few_shots.py --model=ViT-B-32 --clip-backend=openclip --checkpoint-root=checkpoints_openclip --logdir=results_openclip/ --partition 10 --subsample 0.25

echo "Rerunning: [openclip] ViT-B-32: Meta-train text-to-coefficient hypernetwork"
python src/learn_text_to_coef.py --model=ViT-B-32 --clip-backend=openclip --save=checkpoints_openclip/ViT-B-32 --hypernetwork-arch=small --text-source=manual --meta-train-datasets=CIFAR10,EuroSAT,DTD,GTSRB,SVHN,Food101 --meta-val-datasets=Caltech101,Flowers102 --meta-epochs=3 --episodes-per-epoch=3 --blockwise-coef

echo "Rerunning: [openclip] ViT-B-32: Meta-train multi-modal hypernetwork"
python src/learn_multimodal_to_coef.py --model=ViT-B-32 --clip-backend=openclip --save=checkpoints_openclip/ViT-B-32 --hypernetwork-arch=small --fusion-mode=concat --num-shots=4 --image-pooling=mean --text-input-mode=dataset --variable-shots --meta-train-datasets=CIFAR10,EuroSAT,DTD,GTSRB,SVHN,Food101 --meta-val-datasets=Caltech101,Flowers102 --meta-epochs=3 --episodes-per-epoch=3 --blockwise-coef

