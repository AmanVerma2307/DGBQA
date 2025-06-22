#!/bin/bash

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-64-512_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-64-512_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-64-512_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-64-512_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-64-512_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-64-512_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-64-512_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-64-512_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-64-512_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-64-512_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-64-512_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 64 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-64-512_2-2pt5_soli'


python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-128-512_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-128-512_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-128-512_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-128-512_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-128-512_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-128-512_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-128-512_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-128-512_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-128-512_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-128-512_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-128-512_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-128-512_2-2pt5_soli'


python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-256-512_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-256-512_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-246-512_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-256-512_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-256-512_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-256-512_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-256-512_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-256-512_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-256-512_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-256-512_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-256-512_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 512 --exp_name 'dgbqa_res3dvivit-2heads-256-512_2-2pt5_soli'






python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-128-1024_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-128-1024_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-128-1024_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-128-1024_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-128-1024_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-128-1024_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-128-1024_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-128-1024_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-128-1024_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-128-1024_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-128-1024_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 128 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-128-1024_2-2pt5_soli'


python './trainer.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-256-1024_2-0_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-256-1024_2-0_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-246-1024_2-pt005_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.005 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-256-1024_2-pt005_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-256-1024_2-pt05_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.05 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-256-1024_2-pt05_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-256-1024_2-pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 0.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-256-1024_2-pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-256-1024_2-1pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 1.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-256-1024_2-1pt5_soli'

python './trainer.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-256-1024_2-2pt5_soli'
python './tester.py' --lambda_id 2.0 --lambda_icgd 2.5 --num_epochs 150 --device 'cuda:0' --res3dvivit_heads 2 --d_model 256 --dff 1024 --exp_name 'dgbqa_res3dvivit-2heads-256-1024_2-2pt5_soli'