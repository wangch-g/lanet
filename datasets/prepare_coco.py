import os
import argparse

def prepare_coco(args):
    train_file = open(os.path.join(args.saved_dir, args.saved_txt), 'w')
    dirs = os.listdir(args.raw_dir)

    for file in dirs:
        # Write training files
        train_file.write('%s\n' % (file))

    print('Data Preparation Finished.')

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="coco prepareing.")
    arg_parser.add_argument('--dataset', type=str, default='coco',
                             help='')
    arg_parser.add_argument('--raw_dir', type=str, default='',
                             help='')
    arg_parser.add_argument('--saved_dir', type=str, default='',
                             help='')
    arg_parser.add_argument('--saved_txt', type=str, default='train2017.txt',
                             help='')
    args = arg_parser.parse_args() 

    prepare_coco(args)