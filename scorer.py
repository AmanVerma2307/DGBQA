######## Importing libraries
import wandb
import argparse
import numpy as np
from _scorer.DGBQA_Score import gbqa_delta_dist_compute
from _scorer.tfICGDScore import *
from _scorer.RankDeviation import avg_rank_deviation
from _scorer.AcceptanceScore import acceptance_score
from _scorer.PatternMatchDistance import pattern_match_dist
from _scorer.AcceptanceScoreComparison import acceptance_score_comp

####### Model Arguments and Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name",
                    type=str,
                    help="Name of the Experiment being run, will be used saving the model and correponding outputs")
parser.add_argument('--filePath',
                    type=str,
                    default='',
                    help="Path to the saver file")
parser.add_argument('--init',
                    type=int,
                    default=0,
                    help="If true, then the metric writes the measure titles")

args = parser.parse_args()

# api = wandb.Api()
# runs = api.runs(path="eez227536-iit-delhi/dgbqaCodebase")

# for i in runs:
#   if(str(i.name) == args.exp_name):
#      runId = i.id
#      break
  
# run = wandb.init(id=runId,resume=True)

####### Score estimation

##### Defining Essentials
gesture_list = ['Pinch index','Palm tilt','Finger slider','Pinch pinky','Slow swipe','Fast swipe','Push','Pull','Finger rub','Circle','Palm hold']
num_subjects = 10
num_gestures = 11
dgbqa_score = []
Test_Embeddings = np.load('./embeddings/'+str(args.exp_name)+'.npz')['arr_0']
y_dev = np.load('./data/soli/y_dev_DGBQA-Seen_SOLI.npz',allow_pickle=True)['arr_0']
y_dev_id = np.load('./data/soli/y_dev_id_DGBQA-Seen_SOLI.npz',allow_pickle=True)['arr_0']

##### DGBQA Score
for g_id, gesture_curr in enumerate(gesture_list):
    print('==============================================')
    dgbqa_score_curr, d_c_star_curr, d_cs_curr, dgbqa_score_wo_curr = gbqa_delta_dist_compute(Test_Embeddings,g_id,num_subjects,y_dev,y_dev_id)
    dgbqa_score.append(dgbqa_score_curr)
    print('GBQA Delta Distance for '+str(gesture_curr)+' = '+str(dgbqa_score_curr))  

dgbqa_score = np.array(dgbqa_score) # Array Formation
dgbqa_score = (dgbqa_score - np.mean(dgbqa_score))/np.std(dgbqa_score) # Mean Normalization
dgbqa_score = dgbqa_score/np.linalg.norm(dgbqa_score) # L2-Normalization

##################################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
### EER-Processing
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
##################################################################################################################

####### EER-Processing
e = [15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63]
e = np.array(e)
e_prime = 100 - np.array(e)
e_prime = (e_prime - np.mean(e_prime))/np.std(e_prime)
e_prime = e_prime/np.linalg.norm(e_prime)
G = 11
I = 10

##################################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
### Feature-Space Scores
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
##################################################################################################################

####### CGID and Decorr-CGID Score
C_I, C_D = CGID_Score_Calculator(Test_Embeddings,y_dev)

##################################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
### Comparison Scores    
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
##################################################################################################################

####### DGBQA-Score
###### General
beta = 0.75
alpha = 2
nu = 1
rank_dev = avg_rank_deviation(e,dgbqa_score,num_gestures)
Ar = acceptance_score(dgbqa_score,e_prime,11,normalizer=False,relevance=False)
relevance = acceptance_score(dgbqa_score,e_prime,11,normalizer=False,relevance=True)
d = pattern_match_dist(dgbqa_score,e_prime,11)
d_metric = (np.log2(2+nu*d)**(-1/alpha))
O_prime = np.exp(-beta*C_D) # Orthogonal Penalty
Ar_star = Ar*d_metric
Ar_star_plusplus = Ar_star*O_prime
Ar_max = acceptance_score(dgbqa_score,e_prime,11,normalizer=True,relevance=False)
nAr = Ar/Ar_max
nAr_star = Ar_star/Ar_max
nAr_star_plusplus = Ar_star_plusplus/Ar_max
Ar_comp = acceptance_score_comp(dgbqa_score,e_prime,11)*d_metric

print('Rank Deviation: '+str(rank_dev))
print('Relevance: '+str(relevance))
print('Ar: '+str(Ar))
print('d: '+str(d))
print('d_metric: '+str(d_metric))
print('O_prime: '+str(O_prime))
print('CGID Score: '+str(round(C_I,3)))              
print('CGID Score Decorrelated: '+str(round(C_D,3)))
print('Ar_star(Ar*d_metric): '+str(Ar_star))
print('Ar_star_++(Ar*d_metric*O_prime): '+str(Ar_star_plusplus))
print('Ar_max: '+str(Ar_max))
print('nAr: '+str(nAr))
print('nAr_star_++: '+str(nAr_star_plusplus))
print('Ar_comp: '+str(Ar_comp))

measure = ['model','r','R','Psi','Cd','nAr*']
measureVal = [str(args.exp_name),
              str(round(rank_dev,4)),
              str(round(relevance,4)),
              str(round(d,4)),
              str(round(C_I,4)),
              str(round(nAr_star_plusplus,4))]

# run.summary['r'] = rank_dev
# run.summary['Relevance'] = relevance
# run.summary['Psi'] = d
# run.summary['Cd'] = C_I
# run.summary['Ar'] = Ar
# run.summary['nAr*'] = nAr_star_plusplus
# run.summary['Ar_comp'] = Ar_comp

# run.summary.update()

if(args.init == 1): # True: First writing
    
    scoreFile = open('./scoreFiles/'+args.filePath+'.txt','w')
    for item_idx, item in enumerate(measure):
        if(item_idx == 0):
            scoreFile.write(str(item)+'                             ')
        elif(item_idx == 1):
            scoreFile.write(str(item)+'             ')
        elif(item_idx > 1 and item_idx <= 4):
            scoreFile.write(str(item)+'       ')
        else:
            scoreFile.write(str(item)+"\n")

    for item_idx, item in enumerate(measureVal):
        if(item_idx <= 4):
            scoreFile.write(str(item)+'     ')
        else:
            scoreFile.write(str(item)+"\n")

if(args.init == 0):
    scoreFile = open('./scoreFiles/'+args.filePath+'.txt','a')
    for item_idx, item in enumerate(measureVal):
        if(item_idx <= 4):
            scoreFile.write(str(item)+'    ')
        else:
            scoreFile.write(str(item)+"\n")
