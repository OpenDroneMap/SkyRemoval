import argparse
from skyfilter import SkyFilter
import global_vars as gv

parser = argparse.ArgumentParser(description='SkyRemoval')
parser.add_argument('--model', type=str, default=gv.default_model_url, help='Model path, can be a URL or a file path')
parser.add_argument('--ignore_cache', action='store_true', help='Ignore cache when downloading model')
parser.add_argument('--width', type=int, default=384, help='Trained model input width')
parser.add_argument('--height', type=int, default=384, help='Trained Model input height')
parser.add_argument('source', type=str, help='Source image path, can be a single image or a folder')
parser.add_argument('dest', type=str, help='Destination folder path')


if __name__ == '__main__':

    print('\n *** SkyRemoval - %s ***\n' % gv.version)

    args = parser.parse_args()

    filter = SkyFilter(args)
    filter.run(args.source, args.dest)
    


