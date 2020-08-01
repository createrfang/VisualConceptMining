import Augmentor
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='Data Augmentor of VCM')

    parser.add_argument('src', metavar='DIR', help='path to dataset')
    parser.add_argument('--out_num', default=2e5, type=int)
    return parser.parse_args()

def main(args):
    path_to_data = args.src
    outputdir = path_to_data+'_aug'

    # Create a pipeline
    p = Augmentor.Pipeline(path_to_data,output_directory=outputdir)

    p.flip_left_right(probability=0.4)

    # Now we add a vertical flip operation to the pipeline:
    p.flip_top_bottom(probability=0.8)

    # Add a rotate90 operation to the pipeline:
    p.rotate(probability=0.1, max_left_rotation=20, max_right_rotation=20)
    p.scale(probability=0.8, scale_factor=2.0)

    p.sample(args.out_num)


if __name__=="__main__":
    args = parse_args()
    main(args)