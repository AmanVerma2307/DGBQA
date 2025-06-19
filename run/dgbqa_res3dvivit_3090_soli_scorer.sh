#!/bin/bash

conda activate test_1

python './scorer.py' --exp_name dgbqa_res3dvivit_pt5-0_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 1
python './scorer.py' --exp_name dgbqa_res3dvivit_pt5-pt005_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_pt5-pt1_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_pt5-1pt5_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0

python './scorer.py' --exp_name dgbqa_res3dvivit_1-0_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_1-pt005_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_1-pt1_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_1-2_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0

python './scorer.py' --exp_name dgbqa_res3dvivit_1pt5-0_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_1pt5-pt005_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_1pt5-pt5_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_1pt5-2pt5_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0

python './scorer.py' --exp_name dgbqa_res3dvivit_2-0_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_2-pt05_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_2-pt5_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_2-2pt5_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0

python './scorer.py' --exp_name dgbqa_res3dvivit_2pt5-0_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_2pt5-pt005_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_2pt5-pt05_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_2pt5-1_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_2pt5-2pt5_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0

python './scorer.py' --exp_name dgbqa_res3dvivit_3-0_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_3-pt005_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_3-pt5_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_3-2pt5_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
python './scorer.py' --exp_name dgbqa_res3dvivit_3-3_3090_soli --filePath dgbqa_res3dvivit_3090_soli --init 0
