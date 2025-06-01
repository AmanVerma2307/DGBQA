#!/bin/bash

python './trainer.py' --lambda_id 1.5 --lambda_icgd 1.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-1pt5_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 1.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-1pt5_soli'

python './trainer.py' --lambda_id 1.5 --lambda_icgd 2.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-2_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 2.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-2_soli'

python './trainer.py' --lambda_id 1.5 --lambda_icgd 2.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-2pt5_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 2.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-2pt5_soli'
