
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class Vectorizer:

    def __init__ (self, ngram_count=2, type='ngram'):
        self.n = ngram_count
        self.type = type
        if self.type == 'tfidf':
            self.vec = TfidfVectorizer(ngram_range=(self.n, self.n), norm=None, min_df=5, max_df=0.7)
        elif self.type == 'ngram':
            self.vec = CountVectorizer(ngram_range=(self.n, self.n))
            #self.vec = CountVectorizer()
        else:
            raise Exception ('type is not valid.')

    def get_features(self):
        return self.vec.get_feature_names()

    def vectorize(self, samples, to_array=True):
        result = self.vec.fit_transform(samples)
        if to_array:
            return result.toarray()
        return result

