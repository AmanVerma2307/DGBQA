#!/bin/bash

python './trainer.py' --lambda_id 0.5 --lambda_icgd 1.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-1pt5_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 1.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-1pt5_soli'

python './trainer.py' --lambda_id 0.5 --lambda_icgd 2.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-2_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 2.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-2_soli'

python './trainer.py' --lambda_id 0.5 --lambda_icgd 2.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-2pt5_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 2.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-2pt5_soli'



python './trainer.py' --lambda_id 1 --lambda_icgd 1.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-1pt5_soli'
python './tester.py' --lambda_id 1 --lambda_icgd 1.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-1pt5_soli'

python './trainer.py' --lambda_id 1 --lambda_icgd 2.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-2_soli'
python './tester.py' --lambda_id 1 --lambda_icgd 2.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-2_soli'

python './trainer.py' --lambda_id 1 --lambda_icgd 2.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-2pt5_soli'
python './tester.py' --lambda_id 1 --lambda_icgd 2.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-2pt5_soli'



python './trainer.py' --lambda_id 2 --lambda_icgd 1.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-1pt5_soli'
python './tester.py' --lambda_id 2 --lambda_icgd 1.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-1pt5_soli'

python './trainer.py' --lambda_id 2 --lambda_icgd 2.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-2_soli'
python './tester.py' --lambda_id 2 --lambda_icgd 2.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-2_soli'

python './trainer.py' --lambda_id 2 --lambda_icgd 2.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-2pt5_soli'
python './tester.py' --lambda_id 2 --lambda_icgd 2.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-2pt5_soli'



python './trainer.py' --lambda_id 2.5 --lambda_icgd 1.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-1pt5_soli'
python './tester.py' --lambda_id 2.5 --lambda_icgd 1.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-1pt5_soli'

python './trainer.py' --lambda_id 2.5 --lambda_icgd 2.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-2_soli'
python './tester.py' --lambda_id 2.5 --lambda_icgd 2.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-2_soli'

python './trainer.py' --lambda_id 2.5 --lambda_icgd 2.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-2pt5_soli'
python './tester.py' --lambda_id 2.5 --lambda_icgd 2.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-2pt5_soli'



python './trainer.py' --lambda_id 3 --lambda_icgd 0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-0_soli'
python './tester.py' --lambda_id 3 --lambda_icgd 0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-0_soli'

python './trainer.py' --lambda_id 3 --lambda_icgd 1e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-pt001_soli'
python './tester.py' --lambda_id 3 --lambda_icgd 1e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-pt001_soli'

python './trainer.py' --lambda_id 3 --lambda_icgd 5e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-pt005_soli'
python './tester.py' --lambda_id 3 --lambda_icgd 5e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-pt005_soli'

python './trainer.py' --lambda_id 3 --lambda_icgd 1e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-pt01_soli'
python './tester.py' --lambda_id 3 --lambda_icgd 1e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-pt01_soli'

python './trainer.py' --lambda_id 3 --lambda_icgd 5e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-pt05_soli'
python './tester.py' --lambda_id 3 --lambda_icgd 5e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-pt05_soli'

python './trainer.py' --lambda_id 3 --lambda_icgd 0.1 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-pt1_soli'
python './tester.py' --lambda_id 3 --lambda_icgd 0.1 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-pt1_soli'

python './trainer.py' --lambda_id 3 --lambda_icgd 0.5 --num_epochs 200 --exp_name 'dgbqa_res3dvivit_3-pt5_soli'
python './tester.py' --lambda_id 3 --lambda_icgd 0.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-pt5_soli'

python './trainer.py' --lambda_id 3 --lambda_icgd 1.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-1_soli'
python './tester.py' --lambda_id 3 --lambda_icgd 1.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-1_soli'

python './trainer.py' --lambda_id 3 --lambda_icgd 1.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-1pt5_soli'
python './tester.py' --lambda_id 3 --lambda_icgd 1.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-1pt5_soli'

python './trainer.py' --lambda_id 3 --lambda_icgd 2.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-2_soli'
python './tester.py' --lambda_id 3 --lambda_icgd 2.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-2_soli'

python './trainer.py' --lambda_id 3 --lambda_icgd 2.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-2pt5_soli'
python './tester.py' --lambda_id 3 --lambda_icgd 2.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-2pt5_soli'

python './trainer.py' --lambda_id 3 --lambda_icgd 3.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-3_soli'
python './tester.py' --lambda_id 3 --lambda_icgd 3.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_3-3_soli'

