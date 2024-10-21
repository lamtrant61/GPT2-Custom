from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from .utils.common import load_csv_data

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
        X = df['texts'].tolist()
        y = df['labels'].tolist()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.preprocess_data()

    def preprocess_data(self):
        self.vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(self.X_train)
        X_test_tfidf = vectorizer.transform(self.X_test)
        self.x_train = X_train_tfidf
        self.x_test = X_test_tfidf
    
    def train(self, X_train, y_train):
        self.model.summary()
        self.model.fit(self.x_train, self.y_train)

    def dump_model(self):
        joblib.dump(self.model, self.model_name)
        joblib.dump(self.vectorizer, self.vectorizer_name)
    