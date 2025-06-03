#!/bin/bash

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 4 --exp_name 'dgbqa_res3dvivit-4heads_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 4 --exp_name 'dgbqa_res3dvivit-4heads_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 4 --exp_name 'dgbqa_res3dvivit-4heads_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 4 --exp_name 'dgbqa_res3dvivit-4heads_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 4 --exp_name 'dgbqa_res3dvivit-4heads_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 4 --exp_name 'dgbqa_res3dvivit-4heads_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 4 --exp_name 'dgbqa_res3dvivit-4heads_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 4 --exp_name 'dgbqa_res3dvivit-4heads_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 4 --exp_name 'dgbqa_res3dvivit-4heads_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 4 --exp_name 'dgbqa_res3dvivit-4heads_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 4 --exp_name 'dgbqa_res3dvivit-4heads_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 4 --exp_name 'dgbqa_res3dvivit-4heads_2-2pt5_soli'




python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 8 --exp_name 'dgbqa_res3dvivit-8heads_2-0_soli'`
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 8 --exp_name 'dgbqa_res3dvivit-8heads_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 8 --exp_name 'dgbqa_res3dvivit-8heads_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 8 --exp_name 'dgbqa_res3dvivit-8heads_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 8 --exp_name 'dgbqa_res3dvivit-8heads_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 8 --exp_name 'dgbqa_res3dvivit-8heads_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 8 --exp_name 'dgbqa_res3dvivit-8heads_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 8 --exp_name 'dgbqa_res3dvivit-8heads_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 8 --exp_name 'dgbqa_res3dvivit-8heads_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 8 --exp_name 'dgbqa_res3dvivit-8heads_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 8 --exp_name 'dgbqa_res3dvivit-8heads_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 8 --exp_name 'dgbqa_res3dvivit-8heads_2-2pt5_soli'




python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 16 --exp_name 'dgbqa_res3dvivit-16heads_2-0_soli'`
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 16 --exp_name 'dgbqa_res3dvivit-16heads_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 16 --exp_name 'dgbqa_res3dvivit-16heads_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 16 --exp_name 'dgbqa_res3dvivit-16heads_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 16 --exp_name 'dgbqa_res3dvivit-16heads_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 16 --exp_name 'dgbqa_res3dvivit-16heads_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 16 --exp_name 'dgbqa_res3dvivit-16heads_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 16 --exp_name 'dgbqa_res3dvivit-16heads_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 16 --exp_name 'dgbqa_res3dvivit-16heads_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 16 --exp_name 'dgbqa_res3dvivit-16heads_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 16 --exp_name 'dgbqa_res3dvivit-16heads_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:0' --res3dvivit_heads 16 --exp_name 'dgbqa_res3dvivit-16heads_2-2pt5_soli'

