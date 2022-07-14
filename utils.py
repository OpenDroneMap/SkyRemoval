
import glob
import unicodedata
import re
import global_vars as gv
import os

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')
   

def get_cached_url():            
    if not os.path.exists(gv.url_file):
        return None

    with open(gv.url_file, 'r') as f:
        return f.read()
        

def save_cached_url(url):
    with open(gv.url_file, 'w') as f:
        f.write(url)


def find_model_file():

    # Get first file with .onnx extension, pretty naive way
    candidates = glob.glob(os.path.join(gv.default_model_folder, '*.onnx'))
    if len(candidates) == 0:
        raise Exception('No model found (expected at least one file with .onnx extension')
    
    return candidates[0]
