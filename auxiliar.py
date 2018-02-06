import os
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
            print("--- escalando para 100x100 as imagens da pasta: {} ---".format(subdirname))
            
            for filename in os.listdir(subject_path):
                try:
                    if filename.endswith('.png'):                                                                   
                        im = Image.open(os.path.join(subject_path, filename))                        
                        im.thumbnail(tam)                                                
                        #im.convert(mode="L").save("{}segmentacao/{}/{}".format(path, subdirname, filename))     #converte em tons de cinza e depois salva
                        im.save("{}segmentacao/{}/{}".format(path, subdirname, filename))
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