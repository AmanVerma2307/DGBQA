#!/bin/bash

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 16 --d_model 512 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-512-1024_2-1_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 16 --d_model 512 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-512-1024_2-1_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 16 --d_model 512 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-512-1024_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 16 --d_model 512 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-512-1024_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 16 --d_model 512 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-512-1024_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 16 --d_model 512 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-512-1024_2-2pt5_soli'