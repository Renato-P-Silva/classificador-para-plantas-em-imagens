import os
import sys


# Import Matplotlib:
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm

# import facerec modules
from facerec.feature import Fisherfaces
from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.visual import subplot
from facerec.util import minmax_normalize
from facerec.serialization import save_model, load_model
# import numpy, matplotlib and logging
import numpy as np
# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image
import logging


def read_images(path, sz=None):
    c = 0
    X,y = [], []
    tam = (100, 100)
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if filename.endswith('.png'):   
                        print("imagens da pasta:{} ---".format(c))                                         
                        im = Image.open(os.path.join(subject_path, filename))                        
                        im.thumbnail(tam)                        
                        im.convert(mode="L").save("{}/{}".format(subject_path, filename))
                        #resize to given size (if given)                
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError as e:
                    print("I/O error: {0}".format(e))            
                except:
                    print("Unexpected error: {0}".format(sys.exc_info()[0]))
                    raise
            c = c+1
    return [X,y]
    
#caminho dos dados
data1 = "data/"

if __name__ == "__main__":
    # Lendo os dados da imagem. Este deve ser um caminho válido!
    [X,y] = read_images(data1)  