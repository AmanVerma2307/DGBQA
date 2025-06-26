#!/bin/bash

python './scorer.py' --exp_name dgbqa_res3dvivit-2heads-32-4_2-0_soli --filePath dgbqa_res3dvivit-2heads-3264-encoders_soli --init 1
python './scorer.py' --exp_name dgbqa_res3dvivit-2heads-32-4_2-pt5_soli --filePath dgbqa_res3dvivit-2heads-3264-encoders_soli --init 0

python './scorer.py' --exp_name dgbqa_res3dvivit-2heads-32-8_2-0_soli --filePath dgbqa_res3dvivit-2heads-3264-encoders_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit-2heads-32-8_2-pt5_soli --filePath dgbqa_res3dvivit-2heads-3264-encoders_soli --init 0

python './scorer.py' --exp_name dgbqa_res3dvivit-2heads-64-4_2-0_soli --filePath dgbqa_res3dvivit-2heads-3264-encoders_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit-2heads-64-4_2-pt5_soli --filePath dgbqa_res3dvivit-2heads-3264-encoders_soli --init 0

python './scorer.py' --exp_name dgbqa_res3dvivit-2heads-64-8_2-0_soli --filePath dgbqa_res3dvivit-2heads-3264-encoders_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit-2heads-64-8_2-pt5_soli --filePath dgbqa_res3dvivit-2heads-3264-encoders_soli --init 0