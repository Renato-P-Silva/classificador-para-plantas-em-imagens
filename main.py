
#bibliotecas
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from skimage.io import imread_collection,imsave
from sklearn.model_selection import train_test_split
from glob import glob
from scipy.stats import randint as sp_randint
import time
from skimage.color import rgb2grey
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA
from skimage.feature import greycomatrix, greycoprops
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.svm import SVC

# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image
import logging


#redimenciona as imagens
def read_images(path, size):
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)            
            print("--- redimensionando para {} as imagens da pasta: {} ---".format(size, subdirname))
            
            for filename in os.listdir(subject_path):
                try:
                    if filename.endswith('.png'):                                                                   
                        im = Image.open(os.path.join(subject_path, filename))                        
                        im.thumbnail(size)                                                
                        #im.convert(mode="L").save("{}segmentacao/{}/{}".format(path, subdirname, filename))     #converte em tons de cinza e depois salva
                        im.save("data/segmentacao/{}/{}".format(subdirname, filename))
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

path = "data/originais/"
size = (100, 100)
start = time.time()
read_images(path, size)
end = time.time()
print("Tempo para o redimensionamento das imagens:",end - start)

#ler as imagens para a base de dados
path = "data/segmentacao/"
classe1 = glob(path + 'BlackGrass/*.png')
classe2 = glob(path + 'Charlock/*.png')
classe3 = glob(path + 'Cleavers/*.png')
classe4 = glob(path + 'Common\ Chickweed/*.png')
classe5 = glob(path + 'Common\ wheat/*.png')
classe6 = glob(path + 'Fat\ Hen/*.png')
classe7 = glob(path + 'Loose\ Silky-bent/*.png')
classe8 = glob(path + 'Maize/*.png')
classe9 = glob(path + 'Scentless\ Mayweed/*.png')
classe10 = glob(path + 'Shepherds\ Purse/*.png')
classe11 = glob(path + 'Small-flowered\ Cranesbill/*.png')
classe12 = glob(path + 'Sugar beet/*.png')

images_cc = imread_collection(classe1)
images_sc = imread_collection(classe2)
images_sc = imread_collection(classe3)
images_sc = imread_collection(classe4)
images_sc = imread_collection(classe5)
images_sc = imread_collection(classe6)
images_sc = imread_collection(classe7)
images_sc = imread_collection(classe8)
images_sc = imread_collection(classe9)
images_sc = imread_collection(classe10)
images_sc = imread_collection(classe11)
images_sc = imread_collection(classe12)

def segmentation(im):
#Recebe uma imagem calcula limiar de otsu e fazer o recorte obdecendo a regiao resultante desse limiar
	grey_image = rgb2grey(im)
	otsu = threshold_otsu(grey_image)
	im_ = grey_image < otsu
	n_branco = np.sum(im_ == 1)
	n_preto = np.sum(im_ == 0)
	if n_branco > n_preto:
		im_ = 1-im_
	label_img = label(im_, connectivity = grey_image.ndim)#detecta regioesnaoconectadas
	props = regionprops(label_img)#calcula propriedade importantes de cada regiao encontrada (ex.area)

	#Convert todas as regioes que possuem um valor de area menor que a
	#maior area em background da imagem
	area = np.asarray([props[i].area for i in range(len(props))]) #area decadaregiaoencontrada
	max_index = np.argmax(area) #indexdamaiorregiao
	for i in range(len(props)):
		if(props[i].area < props[max_index].area):
			label_img[np.where(label_img == i+1)]=0 #regiao menor que a maior eh marcada como background
	#------------------Recorte da regiao de interessse-----------------------
	#Obtendo os limites verticais das imagens segmentadas
	ymin = np.min(np.where(label_img !=0 )[1])
	ymax = np.max(np.where(label_img !=0 )[1])
	imagem_cortada = imagem[:,ymin:ymax,:]
	return imagem_cortada

start = time.time()
for id_im, imagem in enumerate(images_cc):
	im_name = images_cc.files[id_im].split('/')[-1]
	imagem_segmentada = segmentation(imagem)
	imsave(path + im_name, imagem_segmentada)
	
	#print(im_name)
for id_im, imagem in enumerate(images_sc):
	im_name = images_sc.files[id_im].split('/')[-1]
	imagem_segmentada = segmentation(imagem)
	imsave(path + im_name, imagem_segmentada)
	#print(im_name)

end = time.time()
print("Tempo para a segmentacao das imagens:",end - start)

labels = np.concatenate((np.zeros(len(classe1)),np.ones(len(classe2)),np.zeros(len(classe3)),np.ones(len(classe4))\
,np.zeros(len(classe5)),np.ones(len(classe6)),np.zeros(len(classe7)),np.ones(len(classe8)),np.zeros(len(classe9))\
,np.ones(len(classe10)),np.zeros(len(classe11)),np.ones(len(classe12))))

# Extracting Features using GLCM
path_segmentada = "data/segmentacao/"
classe1 = glob(path_segmentada + 'BlackGrass/*.png')
classe2 = glob(path_segmentada + 'Charlock/*.png')
classe3 = glob(path_segmentada + 'Cleavers/*.png')
classe4 = glob(path_segmentada + 'Common\ Chickweed/*.png')
classe5 = glob(path_segmentada + 'Common\ wheat/*.png')
classe6 = glob(path_segmentada + 'Fat\ Hen/*.png')
classe7 = glob(path_segmentada + 'Loose\ Silky-bent/*.png')
classe8 = glob(path_segmentada + 'Maize/*.png')
classe9 = glob(path_segmentada + 'Scentless\ Mayweed/*.png')
classe10 = glob(path_segmentada + 'Shepherds\ Purse/*.png')
classe11 = glob(path_segmentada + 'Small-flowered\ Cranesbill/*.png')
classe12 = glob(path_segmentada + 'Sugar beet/*.png')
images = imread_collection(classe1 + classe2 + classe3 + classe4 + classe5 + classe6 + classe7\
+ classe8 + classe9 + classe10 + classe11 + classe12)

d = 15

features = np.zeros((len(labels),18)) #6 features x 3 color channels
start = time.time()
for id_im, imagem in enumerate(images):
	for id_ch in range(3):
		matrix0 = greycomatrix( imagem[:,:,id_ch], [d], [0], normed = True )
		matrix1 = greycomatrix( imagem[:,:,id_ch], [d], [np.pi/4], normed = True )
		matrix2 = greycomatrix( imagem[:,:,id_ch], [d], [np.pi/2], normed = True )
		matrix3 = greycomatrix( imagem[:,:,id_ch], [d], [3*np.pi/4], normed = True )
		matrix = ( matrix0+matrix1+matrix2+matrix3)/4
		props = np.zeros((6))
		props[0] = greycoprops( matrix, 'contrast' )
		props[1] = greycoprops( matrix, 'dissimilarity' )
		props[2] = greycoprops( matrix, 'homogeneity' )
		props[3] = greycoprops( matrix, 'energy' )
		props[4] = greycoprops( matrix, 'correlation' )
		props[5] = greycoprops( matrix, 'ASM' )
		features[id_im, id_ch*6:(id_ch+1)*6] = props

end=time.time()
print('Tempo para extrair atributos usando GLCM:', end-start)

# Split Data
train = 0.5
test = 1-train
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = test)
print('-------------------------------------------------------')
print("\nConjunto de treino: {0}\nConjunto de teste: {1}\n".format(X_train.size,  y_train.size))
print('-------------------------------------------------------')
# Random Forest Parameter Estimation
def rf_parameter_estimation(xEst, yEst):

	clf = RandomForestClassifier(n_estimators = 20)

	#specify parameter sand distributions to sample from
	hyperparameters={	"n_estimators": range(10, 1000, 50),
						"max_depth":range(1, 100),
						"max_features":sp_randint(1, xEst.shape[1]),
						"min_samples_split":sp_randint(2, xEst.shape[1]),
						"min_samples_leaf":sp_randint(1, xEst.shape[1]),
						"bootstrap":[True, False],
						"criterion":["gini", "entropy"]
					}

	# run randomized search
	n_iter_search = 20
	random_search = RandomizedSearchCV(clf, param_distributions = hyperparameters, n_iter=n_iter_search, scoring = make_scorer(accuracy_score))
	random_search.fit(xEst, yEst)
	report(random_search.cv_results_)
	return random_search.best_params_

#SVM Parameter Estimation
def svm_parameter_estimation(xEst, yEst):

	hyperparameters={'gamma':[1e-1,1e-2,1e-3,1e-4],'C':[1, 10, 100, 1000]}

	clf = SVC(kernel= 'rbf')
	n_iter_search = 8
	random_search = RandomizedSearchCV(clf, param_distributions = hyperparameters,n_iter = n_iter_search,scoring = make_scorer(accuracy_score))
	random_search.fit(xEst, yEst)
	report(random_search.cv_results_)
	return random_search.best_params_

def report(results, n_top=3):
	for i in range(1, n_top + 1):
		candidates = np.flatnonzero(results['rank_test_score'] == i)
		for candidate in candidates:
			print("Model with rank: {0}".format(i))
			print("Mean validation score: {0:.3f}(std: {1:.3f})".format(
				results['mean_test_score'][candidate], 
				results['std_test_score'][candidate]))
			print("Parameters: {0}".format(results['params'][candidate]))

#Classification unsing all features

start = time.time()
parameters = rf_parameter_estimation(X_train, y_train)
c_rf = RandomForestClassifier(**parameters)
c_rf.fit(X_train, y_train)
pred = c_rf.predict(X_test)
acc_rf = accuracy_score(y_test, pred)
end = time.time()
print('\nTempo para classificacao usando Random Forest:', end - start)
print('Random Forest Accuracy:', acc_rf)
print('-------------------------------------------------------')
start = time.time()
parameters = svm_parameter_estimation(X_train, y_train)
c_svm = SVC(**parameters)
c_svm.fit(X_train, y_train)
pred = c_svm.predict(X_test)
acc_svm = accuracy_score(y_test, pred)

end = time.time()
print('\nTempo para classificacao usando SVM:', end - start)
print('Support Vector Machine Accuracy:', acc_svm)
print('-------------------------------------------------------')

"""
#Classification using PCA
def pca(X_train, X_test, y_train, n_comp):

#PCA transformation for using a 'training' set and a 'testing'set
	pca = PCA(n_components = n_comp)
	pca.fit(X_train, y_train)
	transform = pca.transform(X_test)
	return transform

components = [2, 4, 8, 10, 12]
results_rf = np.zeros(5)
results_svm = np.zeros(5)


start = time.time()
for id_comp, comp in enumerate(components):

	print('------------', 'n comp. = ', comp,'------------')

	X_train_pca = pca(X_train, X_train, y_train, comp)
	X_test_pca = pca(X_train, X_test, y_train, comp)
	#RF
	parameters = rf_parameter_estimation(X_train_pca, y_train)
	c_rf = RandomForestClassifier(**parameters)
	c_rf.fit(X_train_pca, y_train)
	pred = c_rf.predict(X_test_pca)
	acc = accuracy_score(y_test, pred)
	results_rf[id_comp] = acc

	print('-------------------------------------------------------')

	parameters = svm_parameter_estimation(X_train_pca, y_train)
	c_svm = SVC(**parameters)
	c_svm.fit(X_train_pca, y_train)
	pred = c_svm.predict(X_test_pca)
	acc = accuracy_score(y_test, pred)
	results_svm[id_comp] = acc

end = time.time()
print('\nTempo para classificacao usando PCA:', end-start)
#Plot do grafico como resultado para os classificadores SVM e RandomForest
plt.style.use('ggplot')
fig = plt.figure(figsize=(10, 5), dpi=400)
ax = plt.subplot(111)
ax.plot(range(1, 7), np.concatenate((results_rf, [acc_rf]), axis=0), marker='D', linestyle=':', label = 'Random Forest')
ax.plot(range(1, 7), np.concatenate((results_svm, [acc_svm]), axis=0), marker='D',linestyle=':',label='SVM')
ax.set_xlim([0, 7])
ax.set_xlabel('PCA Components')
ax.set_ylabel('Accuracy')
ax.set_xticks(range(1,7))
ax.set_xticklabels(['2','4','8','10','12','NoPCA'])
ax.set_ylim([0.5,1])
ax.set_title('Helmets')
egend = ax.legend(loc='uppercenter', shadow=True)
"""