#!/bin/bash

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_1-0_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_1-0_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_1-pt005_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_1-pt005_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_1-pt05_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_1-pt05_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_1-pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_1-pt5_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_1-1_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_1-1_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_1-1pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_1-1pt5_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_1-2pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_1-2pt5_soli'


python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_2-1_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_2-1_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamp' --exp_name 'dgbqa_res3dvivit-2heads-64-adamp_2-2pt5_soli'





python './trainer.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_1-0_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_1-0_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_1-pt005_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_1-pt005_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_1-pt05_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_1-pt05_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_1-pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_1-pt5_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_1-1_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_1-1_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_1-1pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_1-1pt5_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_1-2pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_1-2pt5_soli'


python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_2-1_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_2-1_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'nadam' --exp_name 'dgbqa_res3dvivit-2heads-64-nadam_2-2pt5_soli'





python './trainer.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_1-0_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_1-0_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_1-pt005_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_1-pt005_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_1-pt05_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_1-pt05_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_1-pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_1-pt5_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_1-1_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_1-1_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_1-1pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_1-1pt5_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_1-2pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_1-2pt5_soli'


python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_2-1_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_2-1_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'radam' --exp_name 'dgbqa_res3dvivit-2heads-64-radam_2-2pt5_soli'






python './trainer.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_1-0_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_1-0_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_1-pt005_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_1-pt005_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_1-pt05_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_1-pt05_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_1-pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_1-pt5_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_1-1_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_1-1_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_1-1pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_1-1pt5_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_1-2pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_1-2pt5_soli'


python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_2-1_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_2-1_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'novograd' --exp_name 'dgbqa_res3dvivit-2heads-64-novograd_2-2pt5_soli'





python './trainer.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_1-0_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_1-0_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_1-pt005_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_1-pt005_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_1-pt05_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_1-pt05_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_1-pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_1-pt5_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_1-1_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_1-1_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_1-1pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_1-1pt5_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_1-2pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_1-2pt5_soli'


python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_2-1_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_2-1_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adafactor' --exp_name 'dgbqa_res3dvivit-2heads-64-adafactor_2-2pt5_soli'




python './trainer.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_1-0_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_1-0_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_1-pt005_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_1-pt005_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_1-pt05_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_1-pt05_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_1-pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_1-pt5_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_1-1_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_1-1_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_1-1pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_1-1pt5_soli'

python './trainer.py' --lambda_id 1.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_1-2pt5_soli'
python './tester.py' --lambda_id 1.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_1-2pt5_soli'


python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_2-1_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_2-1_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --optimizer 'adamw' --exp_name 'dgbqa_res3dvivit-2heads-64-adamw_2-2pt5_soli'