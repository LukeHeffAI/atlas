#!/bin/bash
# Auto-generated script to rerun failed experiments
# Generated: Thu 26 Mar 2026 02:36:16 AM ACDT
set -uo pipefail

echo "Rerunning: [clip] ViT-B-32: Evaluate single-task (none)"
python src/eval_single_task.py --model=ViT-B-32 --clip-backend=clip --checkpoint-root=checkpoints --finetuning-mode=none

echo "Rerunning: [clip] ViT-B-32: Evaluate single-task (standard)"
python src/eval_single_task.py --model=ViT-B-32 --clip-backend=clip --checkpoint-root=checkpoints --finetuning-mode=standard

echo "Rerunning: [clip] ViT-B-32: Learn task negation"
python src/learn_task_negation.py --model=ViT-B-32 --clip-backend=clip --checkpoint-root=checkpoints --blockwise-coef

echo "Rerunning: [clip] ViT-B-32: Learn task addition"
python src/learn_task_addition.py --model=ViT-B-32 --clip-backend=clip --checkpoint-root=checkpoints --blockwise-coef

echo "Rerunning: [clip] ViT-B-32: Test-time adaptation (UFM)"
python src/learn_ufm.py --model=ViT-B-32 --clip-backend=clip --checkpoint-root=checkpoints --logdir=results/ --blockwise-coef

echo "Rerunning: [clip] ViT-B-32: Meta-train text-to-coefficient hypernetwork"
python src/learn_text_to_coef.py --model=ViT-B-32 --clip-backend=clip --save=checkpoints/ViT-B-32 --hypernetwork-arch=small --text-source=manual --meta-train-datasets=CIFAR10,EuroSAT,DTD,GTSRB,SVHN,Food101 --meta-val-datasets=Caltech101,Flowers102 --meta-epochs=3 --episodes-per-epoch=3 --blockwise-coef

echo "Rerunning: [clip] ViT-B-32: Meta-train multi-modal hypernetwork"
python src/learn_multimodal_to_coef.py --model=ViT-B-32 --clip-backend=clip --save=checkpoints/ViT-B-32 --hypernetwork-arch=small --fusion-mode=concat --num-shots=4 --image-pooling=mean --text-input-mode=dataset --variable-shots --meta-train-datasets=CIFAR10,EuroSAT,DTD,GTSRB,SVHN,Food101 --meta-val-datasets=Caltech101,Flowers102 --meta-epochs=3 --episodes-per-epoch=3 --blockwise-coef

echo "Rerunning: [openclip] ViT-B-32: Fine-tune on all datasets"
python src/finetune.py --model=ViT-B-32 --clip-backend=openclip --checkpoint-root=checkpoints_openclip

echo "Rerunning: [openclip] ViT-B-32: Evaluate single-task (none)"
python src/eval_single_task.py --model=ViT-B-32 --clip-backend=openclip --checkpoint-root=checkpoints_openclip --finetuning-mode=none

echo "Rerunning: [openclip] ViT-B-32: Evaluate single-task (standard)"
python src/eval_single_task.py --model=ViT-B-32 --clip-backend=openclip --checkpoint-root=checkpoints_openclip --finetuning-mode=standard

echo "Rerunning: [openclip] ViT-B-32: Learn task negation"
python src/learn_task_negation.py --model=ViT-B-32 --clip-backend=openclip --checkpoint-root=checkpoints_openclip --blockwise-coef

echo "Rerunning: [openclip] ViT-B-32: Learn task addition"
python src/learn_task_addition.py --model=ViT-B-32 --clip-backend=openclip --checkpoint-root=checkpoints_openclip --blockwise-coef

echo "Rerunning: [openclip] ViT-B-32: Test-time adaptation (UFM)"
python src/learn_ufm.py --model=ViT-B-32 --clip-backend=openclip --checkpoint-root=checkpoints_openclip --logdir=results_openclip/ --blockwise-coef

echo "Rerunning: [openclip] ViT-B-32: Meta-train text-to-coefficient hypernetwork"
python src/learn_text_to_coef.py --model=ViT-B-32 --clip-backend=openclip --save=checkpoints_openclip/ViT-B-32 --hypernetwork-arch=small --text-source=manual --meta-train-datasets=CIFAR10,EuroSAT,DTD,GTSRB,SVHN,Food101 --meta-val-datasets=Caltech101,Flowers102 --meta-epochs=3 --episodes-per-epoch=3 --blockwise-coef

echo "Rerunning: [openclip] ViT-B-32: Meta-train multi-modal hypernetwork"
python src/learn_multimodal_to_coef.py --model=ViT-B-32 --clip-backend=openclip --save=checkpoints_openclip/ViT-B-32 --hypernetwork-arch=small --fusion-mode=concat --num-shots=4 --image-pooling=mean --text-input-mode=dataset --variable-shots --meta-train-datasets=CIFAR10,EuroSAT,DTD,GTSRB,SVHN,Food101 --meta-val-datasets=Caltech101,Flowers102 --meta-epochs=3 --episodes-per-epoch=3 --blockwise-coef

