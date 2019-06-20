from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import exceptions
from .serializers import validatePredictRequest
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier

from selenium import webdriver


class Index(APIView):
    def get(self, request):
        return render(request=request, template_name="home/index.html")


class Predict(APIView):
    def post(self, request):
        predict_data = validatePredictRequest(data=request.data)
        if predict_data.is_valid():
            url = predict_data.data['url']

            ### fetch website home page text data

            browser = webdriver.PhantomJS()
            try:
                browser.get(url)
                test_data = browser.find_element_by_tag_name("body").text.encode('utf-8').strip()
            except:
                return Response({"status": False, "error": {"url": ["Enter a valid URL."]}}, status=status.HTTP_200_OK)
            browser.close()
            if len(test_data) == 0:
                return Response({"status": False, "error": {"url": ["Enter a valid URL."]}}, status=status.HTTP_200_OK)

            bunch = load_files('./3cat_websites', encoding='latin-1')
            X_train, X_test, y_train, y_test = train_test_split(bunch.data, bunch.target, test_size=0)
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
            X_train = vectorizer.fit_transform(X_train)
            X_test = vectorizer.transform([test_data])

            algos = {
                "NearestCentroid": NearestCentroid(),
                "RidgeClassifier": RidgeClassifier(),
                "Perceptron": Perceptron(max_iter=50),
                "PassiveAggressiveClassifier": PassiveAggressiveClassifier(max_iter=50),
                "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=10),
                "RandomForestClassifier": RandomForestClassifier(n_estimators=100),
                "LinearSVC": LinearSVC(penalty="l1", dual=False, tol=1e-3),
                "SGDClassifier": SGDClassifier(alpha=.0001, max_iter=50),
                "MultinomialNB": MultinomialNB(alpha=.01),
                "BernoulliNB": BernoulliNB(alpha=.01),
            }
            results = {}
            for algo in list(algos.keys()):
                algos[algo].fit(X_train, y_train)
                pred = algos[algo].predict(X_test)
                score=algos[algo].score(X_train, y_train)
                result = bunch.target_names[pred[0]]
                results[algo] = [result,score]

            return Response(
                {"status": True,
                 "result": results,
                 "raw_data": test_data
                 },
                status=status.HTTP_200_OK)
        else:
            return Response({"status": False, "error": predict_data.errors})
