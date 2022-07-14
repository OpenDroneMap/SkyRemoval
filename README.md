# SkyRemoval

The purpose of this tool is to create sky masks to improve photogrammetric reconstruction.

## Example

![img00471](https://user-images.githubusercontent.com/7868983/177582259-561c1e3a-529e-48a9-a122-2ca5b57a89d8.jpg)
![img00471_mask](https://user-images.githubusercontent.com/7868983/177581964-1f36ca2f-2d52-40e1-b4c2-6869b34ef2fc.png)

## Getting started

```
git clone https://github.com/OpenDroneMap/SkyRemoval.git
cd SkyRemoval
pip install -r requirements.txt
```

Usage:

```
python skyremoval.py source dest
```

It will automatically download the pre-trained model and run the processing. 

`source` can be a folder or a single image file. `dest` should be a folder.

## Parameters

```
usage: skyremoval.py [-h] [--model MODEL] [--ignore_cache] [--in_size_w IN_SIZE_W] [--in_size_h IN_SIZE_H] source dest

SkyRemoval

positional arguments:
  source                Source image path, can be a single image or a folder
  dest                  Destination folder path

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model path, can be a URL or a file path
  --ignore_cache        Ignore cache when downloading model
  --in_size_w IN_SIZE_W
                        Trained model input width
  --in_size_h IN_SIZE_H
                        Trained Model input height
```

## Using CUDA

If you want to speed up the processing, you can use CUDA. Check out the following link for more information:
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

## Reference

The fast guided filter implementation is credited to:
https://github.com/swehrwein/python-guided-filter
