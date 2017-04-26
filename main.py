import sklearn.linear_model
import sklearn.datasets as sd
import sklearn
import sys
import time
import numpy
import cPickle as pickle
import pandas as pd
import PIL.Image
import sklearn.externals.joblib as modelsave
import sklearn.ensemble
import os

def min(x, y):
    if x > y : return y
    else: return x


if __name__ == '__main__':
    # #get pos&neg pic number
    # print "now get picture number"
    # posPicFilepath = "/Users/yukai/Desktop/faceData/positive/"
    # negPicFilepath = "/Users/yukai/Desktop/faceData/negative/"
    # poiPicNum = 0
    # negPicNum = 0
    #
    # pathDir = os.listdir(posPicFilepath)
    # for allDir in pathDir:
    #     child = os.path.join('%s%s' % (posPicFilepath, allDir))
    #     if (child != posPicFilepath + ".DS_Store"):
    #         poiPicNum += 1
    # pathDir = os.listdir(negPicFilepath)
    # for allDir in pathDir:
    #     child = os.path.join('%s%s' % (negPicFilepath, allDir))
    #     if (child != posPicFilepath + ".DS_Store"):
    #         negPicNum += 1
    # print "positive pic number:",poiPicNum
    # print "negative pic number:",negPicNum
    #
    #get train_x example
    img = PIL.Image.open("test.ppm")
    img = img.resize((86, 86), PIL.Image.ANTIALIAS)
    img_ndarray = numpy.asarray(img, dtype='float64')
    img_ndarray.shape = 1,-1
    # # train_x = numpy.array([img_ndarray[0]] * (poiPicNum + negPicNum))
    # # train_y = numpy.array([1] * (poiPicNum + negPicNum))
    # train_x = numpy.array([img_ndarray[0]] * 10000)
    # train_y = numpy.array([1] * 10000)
    test_x = numpy.array([img_ndarray[0]] * 1)
    #
    # #get train_x train_y
    # print "get train_x and train_y"
    # totalpic = 0
    # pathDir = os.listdir(posPicFilepath)
    # for allDir in pathDir:
    #     child = os.path.join('%s%s' % (posPicFilepath, allDir))
    #     if (child != posPicFilepath + ".DS_Store"):
    #         img = PIL.Image.open(child)
    #         img = img.resize((86,86),PIL.Image.ANTIALIAS)
    #         img_ndarray = numpy.asarray(img,dtype='float64')
    #         img_ndarray.shape = 1,-1
    #         train_x[totalpic] = img_ndarray[0]
    #         train_y[totalpic] = 1
    #         totalpic += 1
    #         if (totalpic > 5000):
    #             break;
    # pathDir = os.listdir(negPicFilepath)
    # for allDir in pathDir:
    #     child = os.path.join('%s%s' % (negPicFilepath, allDir))
    #     if (child != negPicFilepath + ".DS_Store"):
    #         img = PIL.Image.open(child)
    #         img = img.resize((86,86),PIL.Image.ANTIALIAS)
    #         img_ndarray = numpy.asarray(img,dtype='float64')
    #         img_ndarray.shape = 1,-1
    #         train_x[totalpic] = img_ndarray[0]
    #         train_y[totalpic] = 0
    #         totalpic += 1
    #         if (totalpic >= 10000):
    #             break;
    # print "train_x.shape:",train_x.shape
    # print "train_y.shape:", train_y.shape
    # print "total pic:",totalpic
    #
    # #train
    # print "training"
    # model = sklearn.tree.DecisionTreeClassifier(max_depth=9,min_samples_leaf=1)
    # ada=sklearn.ensemble.AdaBoostClassifier(base_estimator=model)
    # ABClassifier = sklearn.ensemble.AdaBoostClassifier(n_estimators = 100)
    # ABModel = ABClassifier.fit(train_x, train_y)
    #
    # #modelSave
    # print "modelsave"
    # modelsave.dump(ABModel,'modelSave/ab.model')

    #model load
    ABModel = modelsave.load("modelSave/ab.model")

    #modelpredict
    print "predict"
    img = PIL.Image.open("yaoming.jpeg")
    img = img.resize((86, 86), PIL.Image.ANTIALIAS)
    print "size:",img.size
    img_ndarray = numpy.asarray(img, dtype='float64')
    print img_ndarray.shape
    img_ndarray.shape = 1,-1
    print test_x.shape
    print img_ndarray.shape
    test_x[0] = img_ndarray[0]
    test_y = ABModel.predict(test_x)
    print "-------------------test img------------------"
    print test_y



    inputImg = PIL.Image.open("yaoming.jpeg")
    width,height = inputImg.size
    originStep = min(width,height)
    currentStep = originStep
    step = originStep / 10

    # box = (170,160,340,333)
    # imgCut = inputImg.crop(box)
    # imgCut.show()
    # img_resize = imgCut.resize((86, 86), PIL.Image.ANTIALIAS)
    # img_resize.show()
    # img_trans = numpy.asarray(img_resize, dtype='float64')
    # img_trans.shape = 1, -1
    # test_x[0] = img_trans[0]
    # test_y = ABModel.predict(test_x)
    # if test_y[0] == 1:
    #     imgCut.show()

    picshow = 0
    while (currentStep * 10 > originStep):
        for i in range(0,height - currentStep + 1,step):
            for j in range(0,width - currentStep + 1,step):
                box = (j,i,j + currentStep,i + currentStep)
                imgCut = inputImg.crop(box)
                img_resize = imgCut.resize((86,86),PIL.Image.ANTIALIAS)
                img_trans = numpy.asarray(img_resize, dtype='float64')
                img_trans.shape = 1, -1
                test_x[0] = img_trans[0]
                test_y = ABModel.predict(test_x)
                if test_y[0] == 1:
                    picshow += 1
                    if picshow < 10:
                        imgCut.show()
        currentStep -= step







    # #boston test
    # print "---------------------start here-------------------------"
    #
    # #get boston house price
    # boston = sd.load_boston();
    # #print boston
    #
    # # Split the dataset with sampleRatio
    # sampleRatio = 0.6
    # n_samples = len(boston.target)
    # print "-------------------boston.target-----------------------"
    # print boston.target
    # #print n_samples
    # sampleBoundary = int(n_samples * sampleRatio)
    #
    # # Shuffle the whole data
    # shuffleIdx = range(n_samples)
    # numpy.random.shuffle(shuffleIdx)
    #
    # # Make the training data
    # train_features = boston.data[shuffleIdx[:sampleBoundary]]
    # print "shuffled data"
    # print shuffleIdx[:sampleBoundary]
    # print "train_features"
    # print train_features
    # train_targets = boston.target[shuffleIdx[:sampleBoundary]]
    #
    # # Make the testing data
    # test_features = boston.data[shuffleIdx[sampleBoundary:]]
    # test_targets = boston.target[shuffleIdx[sampleBoundary:]]
    #
    #
    # # Train
    # linearRegression = sklearn.linear_model.LinearRegression()
    # linearRegression.fit(train_features, train_targets)
    # print "train_x type = ",type(train_features)
    # print "train_x element type", type(train_features[0])
    # print "train_y type = ",type(train_targets)
    # print "train_y element type",type(train_targets[0])
    # # Predict
    # predict_targets = linearRegression.predict(test_features)
    #
    # # Evaluation
    # n_test_samples = len(test_targets)
    # X = range(n_test_samples)
    # error = numpy.linalg.norm(predict_targets - test_targets, ord=1) / n_test_samples
    # print "Ordinary Least Squares (Boston) Error: %.2f" % (error)
    #
    # print "result"
    # print predict_targets
    # print test_targets
    # # Draw