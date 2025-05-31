#!/bin/bash

python './trainer.py' --lambda_id 1 --lambda_icgd 5e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-pt005_soli'
python './tester.py' --lambda_id 1 --lambda_icgd 5e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-pt005_soli'

python './trainer.py' --lambda_id 1 --lambda_icgd 1e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-pt01_soli'
python './tester.py' --lambda_id 1 --lambda_icgd 1e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-pt01_soli'

python './trainer.py' --lambda_id 1 --lambda_icgd 5e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-pt05_soli'
python './tester.py' --lambda_id 1 --lambda_icgd 5e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-pt05_soli'

python './trainer.py' --lambda_id 1 --lambda_icgd 0.1 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-pt1_soli'
python './tester.py' --lambda_id 1 --lambda_icgd 0.1 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-pt1_soli'

python './trainer.py' --lambda_id 1 --lambda_icgd 0.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-pt5_soli'
python './tester.py' --lambda_id 1 --lambda_icgd 0.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-pt5_soli'

python './trainer.py' --lambda_id 1 --lambda_icgd 1.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-1_soli'
python './tester.py' --lambda_id 1 --lambda_icgd 1.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1-1_soli'



python './trainer.py' --lambda_id 0.5 --lambda_icgd 0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-0_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-0_soli'

python './trainer.py' --lambda_id 0.5 --lambda_icgd 1e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-pt001_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 1e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-pt001_soli'

python './trainer.py' --lambda_id 0.5 --lambda_icgd 5e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-pt005_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 5e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-pt005_soli'

python './trainer.py' --lambda_id 0.5 --lambda_icgd 1e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-pt01_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 1e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-pt01_soli'

python './trainer.py' --lambda_id 0.5 --lambda_icgd 5e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-pt05_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 5e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-pt05_soli'

python './trainer.py' --lambda_id 0.5 --lambda_icgd 0.1 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-pt1_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 0.1 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-pt1_soli'

python './trainer.py' --lambda_id 0.5 --lambda_icgd 0.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-pt5_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 0.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-pt5_soli'

python './trainer.py' --lambda_id 0.5 --lambda_icgd 1.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-1_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 1.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_pt5-1_soli'



python './trainer.py' --lambda_id 1.5 --lambda_icgd 0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-0_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-0_soli'

python './trainer.py' --lambda_id 1.5 --lambda_icgd 1e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-pt001_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 1e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-pt001_soli'

python './trainer.py' --lambda_id 1.5 --lambda_icgd 5e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-pt005_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 5e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-pt005_soli'

python './trainer.py' --lambda_id 1.5 --lambda_icgd 1e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-pt01_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 1e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-pt01_soli'

python './trainer.py' --lambda_id 1.5 --lambda_icgd 5e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-pt05_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 5e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-pt05_soli'

python './trainer.py' --lambda_id 1.5 --lambda_icgd 0.1 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-pt1_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 0.1 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-pt1_soli'

python './trainer.py' --lambda_id 1.5 --lambda_icgd 0.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-pt5_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 0.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-pt5_soli'

python './trainer.py' --lambda_id 1.5 --lambda_icgd 1.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-1_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 1.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_1pt5-1_soli'



python './trainer.py' --lambda_id 2 --lambda_icgd 0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-0_soli'
python './tester.py' --lambda_id 2 --lambda_icgd 0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-0_soli'

python './trainer.py' --lambda_id 2 --lambda_icgd 1e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-pt001_soli'
python './tester.py' --lambda_id 2 --lambda_icgd 1e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-pt001_soli'

python './trainer.py' --lambda_id 2 --lambda_icgd 5e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-pt005_soli'
python './tester.py' --lambda_id 2 --lambda_icgd 5e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-pt005_soli'

python './trainer.py' --lambda_id 2 --lambda_icgd 1e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-pt01_soli'
python './tester.py' --lambda_id 2 --lambda_icgd 1e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-pt01_soli'

python './trainer.py' --lambda_id 2 --lambda_icgd 5e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-pt05_soli'
python './tester.py' --lambda_id 2 --lambda_icgd 5e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-pt05_soli'

python './trainer.py' --lambda_id 2 --lambda_icgd 0.1 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-pt01_soli'
python './tester.py' --lambda_id 2 --lambda_icgd 0.1 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-pt01_soli'

python './trainer.py' --lambda_id 2 --lambda_icgd 0.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-pt5_soli'
python './tester.py' --lambda_id 2 --lambda_icgd 0.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-pt5_soli'

python './trainer.py' --lambda_id 2 --lambda_icgd 1.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-1_soli'
python './tester.py' --lambda_id 2 --lambda_icgd 1.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2-1_soli'


python './trainer.py' --lambda_id 2.5 --lambda_icgd 0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-0_soli'
python './tester.py' --lambda_id 2.5 --lambda_icgd 0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-0_soli'

python './trainer.py' --lambda_id 2.5 --lambda_icgd 1e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-pt001_soli'
python './tester.py' --lambda_id 2.5 --lambda_icgd 1e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-pt001_soli'

python './trainer.py' --lambda_id 2.5 --lambda_icgd 5e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-pt005_soli'
python './tester.py' --lambda_id 2.5 --lambda_icgd 5e-3 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-pt005_soli'

python './trainer.py' --lambda_id 2.5 --lambda_icgd 1e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-pt01_soli'
python './tester.py' --lambda_id 2.5 --lambda_icgd 1e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-pt01_soli'

python './trainer.py' --lambda_id 2.5 --lambda_icgd 5e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-pt05_soli'
python './tester.py' --lambda_id 2.5 --lambda_icgd 5e-2 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-pt05_soli'

python './trainer.py' --lambda_id 2.5 --lambda_icgd 0.1 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-pt1_soli'
python './tester.py' --lambda_id 2.5 --lambda_icgd 0.1 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-pt1_soli'

python './trainer.py' --lambda_id 2.5 --lambda_icgd 0.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-pt5_soli'
python './tester.py' --lambda_id 2.5 --lambda_icgd 0.5 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-pt5_soli'

python './trainer.py' --lambda_id 2.5 --lambda_icgd 1.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-1_soli'
python './tester.py' --lambda_id 2.5 --lambda_icgd 1.0 --num_epochs 100 --exp_name 'dgbqa_res3dvivit_2pt5-1_soli'