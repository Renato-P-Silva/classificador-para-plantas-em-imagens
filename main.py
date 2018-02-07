
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

from write_to_file import WriteResults

# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image
import logging

#ARQUIVO PARA A SAÍDA #########################################################
Output = []
###############################################################################

#redimenciona as imagens
def read_images(path, size, OutputPath):
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
                        im.save((OutputPath + "/{}/{}").format(subdirname, filename))
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
read_images(path, size, "data/segmentacao/")
end = time.time()
print("Tempo para o redimensionamento das imagens:",end - start)

#ler as imagens para a base de dados ##########################################
path = "data/segmentacao/"
Classes = list(range(12))

Classes[0] = glob(path + 'BlackGrass/*.png')
Classes[1] = glob(path + 'Charlock/*.png')
Classes[2] = glob(path + 'Cleavers/*.png')
Classes[3] = glob(path + 'Common\ Chickweed/*.png')
Classes[4] = glob(path + 'Common\ wheat/*.png')
Classes[5] = glob(path + 'Fat\ Hen/*.png')
Classes[6] = glob(path + 'Loose\ Silky-bent/*.png')
Classes[7] = glob(path + 'Maize/*.png')
Classes[8] = glob(path + 'Scentless\ Mayweed/*.png')
Classes[9] = glob(path + 'Shepherds\ Purse/*.png')
Classes[10] = glob(path + 'Small-flowered\ Cranesbill/*.png')
Classes[11] = glob(path + 'Sugar\ beet/*.png')

ImagesPerClass = []
for Class in range(12):
	ImagesPerClass.append(imread_collection(Classes[Class]))

###############################################################################


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
for Class in ImagesPerClass:
	for id_im, imagem in enumerate(Class):
		im_name = Class.files[id_im].split('/')[-1]
		imagem_segmentada = segmentation(imagem)
		imsave(path + im_name, imagem_segmentada)



end = time.time()
print("Tempo para a segmentacao das imagens:",end - start)

#IMAGENS DE TESTE #############################################################
size = (100, 100)
start = time.time()
read_images("data/teste/", size, "data/teste_segmentado/")
end = time.time()
print("Tempo para o redimensionamento das imagens: CONJUNTO TESTE",end - start)

#Lê as imagens de teste #######################################################
TestData = glob('data/teste_segmentado/teste/*.png')
TestDataCollection = imread_collection(TestData)


start = time.time()
for id_im, imagem in enumerate(TestDataCollection):
	im_name = TestDataCollection.files[id_im].split('/')[-1]
	imagem_segmentada = segmentation(imagem)
	imsave("data/teste_segmentado/" + im_name, imagem_segmentada)
end = time.time()
print("Tempo para a segmentacao das imagens: CONJUNTO TESTE",end - start)

LabelsTest = np.zeros(len(TestData))

Images_Test = imread_collection(TestData)
d = 15
FeaturesTest = np.zeros((len(LabelsTest), 18))
###############################################################################


start = time.time()
for id_im, imagem in enumerate(Images_Test):
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
		FeaturesTest[id_im, id_ch*6:(id_ch+1)*6] = props

end=time.time()
print('Tempo para extrair atributos usando GLCM:', end-start)




labels = np.concatenate((np.zeros(len(Classes[0])), np.ones(len(Classes[1])),\
np.zeros(len(Classes[2])), np.ones(len(Classes[3])), np.zeros(len(Classes[4])),\
np.ones(len(Classes[5])), np.zeros(len(Classes[6])), np.ones(len(Classes[7])),\
np.zeros(len(Classes[8])), np.ones(len(Classes[9])), np.zeros(len(Classes[10])),\
np.ones(len(Classes[11]))))

images = imread_collection(Classes[0] + Classes[1] + Classes[2] + Classes[3]\
+ Classes[4] + Classes[5] + Classes[6] + Classes[7] + Classes[8] + Classes[9]\
+ Classes[10] + Classes[11])

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

'''
# Split Data
train = 0.5
test = 1-train
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = test)
print('-------------------------------------------------------')
print("\nConjunto de treino: {0}\nConjunto de teste: {1}\n".format(X_train.size,  y_train.size))
print('-------------------------------------------------------')
'''

#GAMBIARRA#####################################################################
# Split Data
train = 1
test = 0
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = test)
print('-------------------------------------------------------')
print("\nConjunto de treino: {0}\nConjunto de teste: {1}\n".format(X_train.size,  X_test.size))
print('-------------------------------------------------------')

train = 0
test = 1
J, XTest, K, YTest = train_test_split(FeaturesTest, LabelsTest, test_size = test)
print('-------------------------------------------------------')
print("\nConjunto de treino: {0}\nConjunto de teste: {1}\n".format(J.size,  XTest.size))
print('-------------------------------------------------------')


#FIM DA GAMBIARRA##############################################################

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
pred = c_rf.predict(XTest)#resultado predito RFC
end = time.time()
print('\nTempo para classificacao usando Random Forest:', end - start)

start = time.time()
parameters = svm_parameter_estimation(X_train, y_train)
c_svm = SVC(**parameters)
c_svm.fit(X_train, y_train)
pred = c_svm.predict(XTest)#resultado predito SVM

end = time.time()
print('\nTempo para classificacao usando SVM:', end - start)

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
