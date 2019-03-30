
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Vectorizer:

    def __init__ (self, ngram_count=2, type='ngram'):
        self.n=ngram_count
        self.type = type
        if type == 'tfidf':
            self.vect = TfidfVectorizer(ngram_range=(self.n, self.n), norm=None)
        if type == 'ngram':
            self.vec= CountVectorizer(ngram_range=(self.n, self.n))
        else:
            raise Exception ('type is not valid.')

    def get_features(self):
        return self.vec.get_feature_names()

    def vectorize(self, samples, to_array=True):
        result = self.vec.fit_transform(samples)
        if to_array:
            return result.toarray()
        return result