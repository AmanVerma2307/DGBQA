#!/bin/bash

python './trainer.py' --lambda_id 0.5 --lambda_icgd 0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_pt5-0_3090_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_pt5-0_3090_soli'

python './trainer.py' --lambda_id 0.5 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_pt5-pt005_3090_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_pt5-pt005_3090_soli'

python './trainer.py' --lambda_id 0.5 --lambda_icgd 0.1 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_pt5-pt1_3090_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 0.1 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_pt5-pt1_3090_soli'

python './trainer.py' --lambda_id 0.5 --lambda_icgd 1.5 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_pt5-1pt5_3090_soli'
python './tester.py' --lambda_id 0.5 --lambda_icgd 1.5 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_pt5-1pt5_3090_soli'



python './trainer.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_1-0_3090_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_1-0_3090_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_1-pt005_3090_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_1-pt005_3090_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.1 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_1-pt1_3090_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.1 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_1-pt1_3090_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 2.0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_1-2_3090_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 2.0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_1-2_3090_soli'



python './trainer.py' --lambda_id 1.5 --lambda_icgd 0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_1pt5-0_3090_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_1pt5-0_3090_soli'

python './trainer.py' --lambda_id 1.5 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_1pt5-pt005_3090_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_1pt5-pt005_3090_soli'

python './trainer.py' --lambda_id 1.5 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_1pt5-pt5_3090_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_1pt5-pt5_3090_soli'

python './trainer.py' --lambda_id 1.5 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_1pt5-2pt5_3090_soli'
python './tester.py' --lambda_id 1.5 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_1pt5-2pt5_3090_soli'




python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2-0_3090_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2-0_3090_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2-pt05_3090_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2-pt05_3090_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2-pt5_3090_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2-pt5_3090_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2-2pt5_3090_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2-2pt5_3090_soli'



python './trainer.py' --lambda_id 2.5 --lambda_icgd 0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2pt5-0_3090_soli'
python './tester.py' --lambda_id 2.5 --lambda_icgd 0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2pt5-0_3090_soli'

python './trainer.py' --lambda_id 2.5 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2pt5-pt005_3090_soli'
python './tester.py' --lambda_id 2.5 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2pt5-pt005_3090_soli'

python './trainer.py' --lambda_id 2.5 --lambda_icgd 0.05 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2pt5-pt05_3090_soli'
python './tester.py' --lambda_id 2.5 --lambda_icgd 0.05 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2pt5-pt05_3090_soli'

python './trainer.py' --lambda_id 2.5 --lambda_icgd 1.0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2pt5-1_3090_soli'
python './tester.py' --lambda_id 2.5 --lambda_icgd 1.0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2pt5-1_3090_soli'

python './trainer.py' --lambda_id 2.5 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2pt5-2pt5_3090_soli'
python './tester.py' --lambda_id 2.5 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_2pt5-2pt5_3090_soli'




python './trainer.py' --lambda_id 3.0 --lambda_icgd 0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_3-0_3090_soli'
python './tester.py' --lambda_id 3.0 --lambda_icgd 0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_3-0_3090_soli'

python './trainer.py' --lambda_id 3.0 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_3-pt005_3090_soli'
python './tester.py' --lambda_id 3.0 --lambda_icgd 0.005 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_3-pt005_3090_soli'

python './trainer.py' --lambda_id 3.0 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_3-pt5_3090_soli'
python './tester.py' --lambda_id 3.0 --lambda_icgd 0.5 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_3-pt5_3090_soli'

python './trainer.py' --lambda_id 3.0 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_3-2pt5_3090_soli'
python './tester.py' --lambda_id 3.0 --lambda_icgd 2.5 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_3-2pt5_3090_soli'

python './trainer.py' --lambda_id 3.0 --lambda_icgd 3.0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_3-3_3090_soli'
python './tester.py' --lambda_id 3.0 --lambda_icgd 3.0 --num_epochs 100 --device 'cuda:1' --exp_name 'dgbqa_res3dvivit_3-3_3090_soli'
