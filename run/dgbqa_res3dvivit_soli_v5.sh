#!/bin/bash

python './trainer.py' --lambda_id 0.5 --lambda_icgd 0 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_pt5-0_1660_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 0 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_pt5-0_1660_soli'

python './trainer.py' --lambda_id 0.5 --lambda_icgd 0.1 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_pt5-pt1_1660_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 0.1 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_pt5-pt1_1660_soli'

python './trainer.py' --lambda_id 0.5 --lambda_icgd 1.5 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_pt5-1pt5_1660_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 1.5 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_pt5-1pt5_1660_soli'



python './trainer.py' --lambda_id 1.5 --lambda_icgd 0 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_1pt5-0_1660_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 0 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_1pt5-0_1660_soli'

python './trainer.py' --lambda_id 1.5 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_1pt5-pt005_1660_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_1pt5-pt005_1660_soli'

python './trainer.py' --lambda_id 1.5 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_1pt5-pt5_1660_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_1pt5-pt5_1660_soli'



python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_2-0_1660_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_2-0_1660_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_2-pt05_1660_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_2-pt05_1660_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_2-pt5_1660_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_2-pt5_1660_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_2-2pt5_1660_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_2-2pt5_1660_soli'



python './trainer.py' --lambda_id 3.0 --lambda_icgd 0 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_3-0_1660_soli'
python './tester.py' --lambda_id 3.0 --lambda_icgd 0 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_3-0_1660_soli'

python './trainer.py' --lambda_id 3.0 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_3-pt005_1660_soli'
python './tester.py' --lambda_id 3.0 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_3-pt005_1660_soli'

python './trainer.py' --lambda_id 3.0 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_3-pt5_1660_soli'
python './tester.py' --lambda_id 3.0 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_3-pt5_1660_soli'

python './trainer.py' --lambda_id 3.0 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_3-2pt5_1660_soli'
python './tester.py' --lambda_id 3.0 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:0' --exp_name 'dgbqa_res3dvivit_3-2pt5_1660_soli'