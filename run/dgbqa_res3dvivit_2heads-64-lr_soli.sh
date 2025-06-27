#!/bin/bash

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt001_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt001_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt001_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt001_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt001_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt001_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt001_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt001_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt001_2-1_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt001_2-1_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt001_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt001_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt001_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt001_2-2pt5_soli'




python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt005_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt005_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt005_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt005_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt005_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt005_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt005_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt005_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 5.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt005_2-5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 5.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt005_2-5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 5.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt005_2-5pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 5.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt005_2-5pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt005_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-3 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt005_2-2pt5_soli'




python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-4 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt0005_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-4 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt0005_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-4 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt0005_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-4 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt0005_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-4 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt0005_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-4 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt0005_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-4 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt0005_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-4 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt0005_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 5.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-4 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt0005_2-5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 5.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-4 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt0005_2-5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 5.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-4 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt0005_2-5pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 5.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-4 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt0005_2-5pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-4 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt0005_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-4 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt0005_2-2pt5_soli'




python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00001_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00001_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00001_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00001_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00001_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00001_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00001_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00001_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 5.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00001_2-5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 5.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00001_2-5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 5.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00001_2-5pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 5.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00001_2-5pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00001_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 1e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00001_2-2pt5_soli'



python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00005_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00005_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00005_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00005_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00005_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00005_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00005_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00005_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 5.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00005_2-5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 5.0 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00005_2-5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 5.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00005_2-5pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 5.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00005_2-5pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00005_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 250 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64  --lr 5e-5 --exp_name 'dgbqa_res3dvivit-2heads-64-lrpt00005_2-2pt5_soli'




