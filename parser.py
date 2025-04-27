import argparse

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        type=str,
                        default='soli',
                        help="Dataset to be used for train/test: [soli,tiny,handLogin,scut]")
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help="Batch size")
    parser.add_argument('--shuffle',
                        type=bool,
                        default=False,
                        help="Shuffling in the dataset")
    parser.add_argument('--multi_gpu',
                        type=bool,
                        default=False,
                        help="If True, then multi GPU training/testing")
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help="Device to be used in single gpu setting")
    parser.add_argument('--model',
                        type=str,
                        default='res3dViViT',
                        help="Model to be used")
    parser.add_argument('--lambda_id',
                        type=float,
                        default=1.0,
                        help="Weight of ID loss")
    parser.add_argument('--lambda_icgd',
                        type=float,
                        default=1.0,
                        help="Weight of ICGD loss")
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help="Number of training epochs")
    parser.add_argument('--exp_name',
                        type=str,
                        help="Name of the experiment")
    
    args = parser.parse_args()
    return args