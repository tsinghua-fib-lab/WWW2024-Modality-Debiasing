import os
import argparse
from utils.quick_start import quick_start, dataset_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'
import faulthandler
faulthandler.enable()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MMGCN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='clothing', help='name of datasets')

    config_dict = {
        'gpu_id':1,
    }

    args, _ = parser.parse_known_args()

    # dataset_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)
    
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


