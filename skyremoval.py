import time
import numpy as np
import cv2
import os
import glob
import argparse
import utils
import urllib.request 
import zipfile
import onnx
import onnxruntime as ort
from guidedfilter import guided_filter

# This version should match the tag in the repository
version = "v1.0.1"
default_model_url = "https://github.com/OpenDroneMap/SkyRemoval/releases/download/%s/model.zip" % version 
default_model_folder = "model"
url_file = os.path.join(default_model_folder, 'url.txt')
interpolation_mode = 'bicubic'
guided_filter_radius, guided_filter_eps = 20, 0.01


# Use GPU if it is available, otherwise CPU
provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"

parser = argparse.ArgumentParser(description='SkyRemoval')
parser.add_argument('--model', type=str, default=default_model_url, help='Model path, can be a URL or a file path')
parser.add_argument('--ignore_cache', action='store_true', help='Ignore cache when downloading model')
parser.add_argument('--width', type=int, default=384, help='Trained model input width')
parser.add_argument('--height', type=int, default=384, help='Trained Model input height')
parser.add_argument('source', type=str, help='Source image path, can be a single image or a folder')
parser.add_argument('dest', type=str, help='Destination folder path')

class SkyFilter():

    def __init__(self, args):

        self.model = args.model
        self.source = args.source
        self.ignore_cache = args.ignore_cache

        self.width, self.height = args.width, args.height

        print(' ?> Using provider %s' % provider)
        self.load_model()

        # Remove trailing slash if present
        if self.source[-1] == '/':
            self.source = self.source[:-1]
        
        self.dest = args.dest

        if os.path.exists(args.dest) is False:
            os.mkdir(args.dest)

    def get_cached_url(self):
            
        if not os.path.exists(url_file):
            return None

        with open(url_file, 'r') as f:
            return f.read()

    def save_cached_url(self, url):
        with open(url_file, 'w') as f:
            f.write(url)

    def load_model(self):
        
        # Check if model is path or url
        if not os.path.exists(self.model):
          
            if not os.path.exists(default_model_folder):
                os.mkdir(default_model_folder)

            if self.ignore_cache:

                print(" ?> We are ignoring the cache")
                self.model = self.get_model(self.model)

            else:

                cached_url = self.get_cached_url()

                if cached_url is None:

                    url = self.model
                    self.model = self.get_model(self.model)
                    self.save_cached_url(url)

                else:
                    
                    if cached_url != self.model:
                        url = self.model
                        self.model = self.get_model(self.model)
                        self.save_cached_url(url)
                    else:
                        self.model = self.find_model_file()

        print(' -> Loading the model')
        onnx_model = onnx.load(self.model)

        # Check the model
        try:
            onnx.checker.check_model(onnx_model)
        except onnx.checker.ValidationError as e:
            print(' !> The model is invalid: %s' % e)
            raise
        else:
            print(' ?> The model is valid!')

        self.session = ort.InferenceSession(self.model, providers=[provider])     


    def get_model(self, url):

        print(' -> Downloading model from: %s' % url)

        dest_file = os.path.join(default_model_folder, utils.slugify(os.path.basename(url)))

        urllib.request.urlretrieve(url, dest_file)

        # Check if model is a zip file
        if os.path.splitext(url)[1].lower() == '.zip':
            print(' -> Extracting model')
            with zipfile.ZipFile(dest_file, 'r') as zip_ref:
                zip_ref.extractall(default_model_folder)
            os.remove(dest_file)

            return self.find_model_file()

        else:
            return dest_file

    def find_model_file(self):

        # Get first file with .onnx extension, pretty naive way
        candidates = glob.glob(os.path.join(default_model_folder, '*.onnx'))
        if len(candidates) == 0:
            raise Exception('No model found (expected at least one file with .onnx extension')
        
        return candidates[0]


    def get_mask(self, img):

        height, width, c = img.shape

        # Resize image to fit the model input
        new_img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        new_img = np.array(new_img, dtype=np.float32)

        # Input vector for onnx model
        input = np.expand_dims(new_img.transpose((2, 0, 1)), axis=0)
        ort_inputs = {self.session.get_inputs()[0].name: input}

        # Run the model
        ort_outs = self.session.run(None, ort_inputs)

        # Get the output
        output = np.array(ort_outs)
        output = output[0][0].transpose((1, 2, 0))
        output = cv2.resize(output, (width, height), interpolation=cv2.INTER_LANCZOS4)
        output = np.array([output, output, output]).transpose((1, 2, 0))
        output = np.clip(output, a_max=1.0, a_min=0.0)

        return self.refine(output, img)        


    def refine(self, pred, img):


        #GF = GuidedFilter(img[:,:,2], guided_filter_radius, guided_filter_eps)
        #refined = GF.filter(pred[:,:,0])

        refined = guided_filter(img[:,:,2], pred[:,:,0], guided_filter_radius, guided_filter_eps)

        #refined = guidedFilter(img[:,:,2], pred[:,:,0], r, eps)

        res = np.clip(refined, a_min=0, a_max=1)
        
        # Convert res to CV_8UC1
        res = np.array(res * 255., dtype=np.uint8)
        
        # Thresholding
        res = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY_INV)[1]
        
        return res
        

    def run_folder(self):

        print(' -> Processing folder ' + self.source)

        img_names = os.listdir(self.source)

        start = time.time()

        # Filter files to only include images
        img_names = [name for name in img_names if os.path.splitext(name)[1] in ['.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff']]

        for idx in range(len(img_names)):
            img_name = img_names[idx]
            print(' -> [%d / %d] processing %s' % (idx+1, len(img_names), img_name))
            self.run_img(os.path.join(self.source, img_name))

        expired = time.time() - start
                
        print('\n ?> Done in %.2f seconds' % expired)
        print(' ?> Elapsed time per image: %.2f seconds' % (expired / len(img_names)))
        print('\n ?> Output saved in ' + self.dest)



    def run_img(self, img_path):

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img / 255., dtype=np.float32)

        mask  = self.get_mask(img)
        
        img_name = os.path.basename(img_path)
        fpath = os.path.join(self.dest, img_name)

        cv2.imwrite(fpath[:-4] + '_mask.png', mask)



    def run(self):

        # Check if source is a directory or a file
        if os.path.isdir(self.source):
            self.run_folder()
        else:
            print(' -> Processing: %s' % self.source)
            start = time.time()
            self.run_img(self.source)
            print(" -> Done in %.2f seconds" % (time.time() - start))
            print(' ?> Output saved in ' + self.dest)



if __name__ == '__main__':

    print('\n *** SkyRemoval - %s ***\n' % version)

    SkyFilter(parser.parse_args()).run()
    


