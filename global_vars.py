import os

# This version should match the tag in the repository
version = "v1.0.6"
default_model_url = "https://github.com/OpenDroneMap/SkyRemoval/releases/download/%s/model.zip" % version 
default_model_folder = "model"
url_file = os.path.join(default_model_folder, 'url.txt')
guided_filter_radius, guided_filter_eps = 20, 0.01