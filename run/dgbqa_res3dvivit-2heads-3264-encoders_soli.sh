#!/bin/bash

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 32 --num_encoders 4 --exp_name 'dgbqa_res3dvivit-2heads-32-4_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 32 --num_encoders 4 --exp_name 'dgbqa_res3dvivit-2heads-32-4_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 32 --num_encoders 4 --exp_name 'dgbqa_res3dvivit-2heads-32-4_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 32 --num_encoders 4 --exp_name 'dgbqa_res3dvivit-2heads-32-4_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 32 --num_encoders 8 --exp_name 'dgbqa_res3dvivit-2heads-32-8_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 32 --num_encoders 8 --exp_name 'dgbqa_res3dvivit-2heads-32-4_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 32 --num_encoders 8 --exp_name 'dgbqa_res3dvivit-2heads-32-8_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 32 --num_encoders 8 --exp_name 'dgbqa_res3dvivit-2heads-32-8_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --num_encoders 4 --exp_name 'dgbqa_res3dvivit-2heads-64-4_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --num_encoders 4 --exp_name 'dgbqa_res3dvivit-2heads-64-4_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --num_encoders 4 --exp_name 'dgbqa_res3dvivit-2heads-64-4_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --num_encoders 4 --exp_name 'dgbqa_res3dvivit-2heads-64-4_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --num_encoders 8 --exp_name 'dgbqa_res3dvivit-2heads-64-8_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --num_encoders 8 --exp_name 'dgbqa_res3dvivit-2heads-64-4_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --num_encoders 8 --exp_name 'dgbqa_res3dvivit-2heads-64-8_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --num_encoders 8 --exp_name 'dgbqa_res3dvivit-2heads-64-8_2-pt5_soli'