#!/bin/bash

python './scorer.py' --exp_name dgbqa_res3dvivit-2heads-512-1024_2-0_soli --filePath dgbqa_res3dvivit-16heads-512-1024_soli --init 1
python './scorer.py' --exp_name dgbqa_res3dvivit-2heads-512-1024_2-pt005_soli --filePath dgbqa_res3dvivit-16heads-512-1024_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit-2heads-512-1024_2-pt05_soli --filePath dgbqa_res3dvivit-16heads-512-1024_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit-2heads-512-1024_2-pt5_soli --filePath dgbqa_res3dvivit-16heads-512-1024_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit-2heads-512-1024_2-1_soli --filePath dgbqa_res3dvivit-16heads-512-1024_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit-2heads-512-1024_2-1pt5_soli --filePath dgbqa_res3dvivit-16heads-512-1024_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit-2heads-512-1024_2-2pt5_soli --filePath dgbqa_res3dvivit-16heads-512-1024_soli --init 0