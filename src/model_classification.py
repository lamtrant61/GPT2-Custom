from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

class Model_SVC:
    def __init__(self, is_init=True):
        self.model_name = './model/svc/svc_model.pkl'
        self.vectorizer_name = './model/svc/tfidf_vectorizer.pkl'
        if is_init:
            self.model = SVC(kernel='linear', C=1, probability=True)
        else:
            self.load_model()

    def load_model(self, filename):
        self.model = joblib.load(self.model_name)
        self.vectorizer = joblib.load(self.vectorizer_name)

    def load_data_train(self, filename):
        df = load_csv_data(filename)
        X = df['text'].tolist()
        y = df['label'].tolist()
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def preprocess_data(self, train_data, test_data):
        self.vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(train_data)
        X_test_tfidf = vectorizer.transform(test_data)
        self.x_train = X_train_tfidf
        self.y_train = y_train
        self.x_test = X_test_tfidf
        self.y_test = y_test
    
    def train(self, X_train, y_train):
        self.model.summary()
        self.model.fit(self.x_train, self.y_train)

    def dump_model(self):
        joblib.dump(self.model, self.model_name)
        joblib.dump(self.vectorizer, self.vectorizer_name)
    