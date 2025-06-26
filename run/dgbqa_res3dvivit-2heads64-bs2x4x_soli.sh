#!/bin/bash

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_1-0_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_1-0_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_1-pt005_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_1-pt005_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_1-pt05_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_1-pt05_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_1-pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_1-pt5_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_1-1_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_1-1_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_1-1pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_1-1pt5_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_1-2pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_1-2pt5_soli'


python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_2-1_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_2-1_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 64 --exp_name 'dgbqa_res3dvivit-2heads-64-bs64_2-2pt5_soli'





python './trainer.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_1-0_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_1-0_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_1-pt005_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_1-pt005_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_1-pt05_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_1-pt05_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_1-pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_1-pt5_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_1-1_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_1-1_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_1-1pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_1-1pt5_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_1-2pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_1-2pt5_soli'


python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_2-1_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_2-1_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --batch_size 128 --exp_name 'dgbqa_res3dvivit-2heads-64-bs128_2-2pt5_soli'