import numpy as np

class NaiveBayesClassification:

    def __init__(self, A, B):
        self.A = A 
        self.B = B 

        #used in both methods
        self.P_word_1 = np.ones(len(self.A[0])) #length = words loaded -2
        self.P_word_0 = np.ones(len(self.A[0])) #length = words loaded -2

        self.P_train_reviews_1 = 0
        self.P_train_reviews_0 = 0
  
    def fit(self):
        
        sum_words_1 = 0 #sum of ALL the words used in a positive review
        sum_words_0 = 0 #sum of ALL the words used in a negative review

        m = len(self.A) #num of reviews (2500)
        n = len(self.A[0]) #words loaded -2

        rows_1 = np.where(self.B==1)[0]
        rows_0 = np.where(self.B==0)[0]

        for j in range(n):
            self.P_word_1[j] = np.sum(self.A[rows_1,j])
            self.P_word_0[j] = np.sum(self.A[rows_0,j])

        for j in range(n):
            sum_words_1 += np.where(self.A[:,j] + self.B == 2)[0].shape[0]
            sum_words_0 += np.where(self.A[:,j] > self.B)[0].shape[0]

        for j in range(n):
            self.P_word_1[j] = self.P_word_1[j]/sum_words_1
            self.P_word_0[j] = self.P_word_0[j]/sum_words_0

        self.P_train_reviews_1 = np.sum(self.B)
        self.P_train_reviews_0 = len(self.B) - self.P_train_reviews_1

        self.P_train_reviews_1 = self.P_train_reviews_1 / len(self.B)#0.5
        self.P_train_reviews_0 = self.P_train_reviews_0  / len(self.B)#0.5

    def predict(self,X):

        length = len(X)
        Y = np.zeros(length)
  
        for i in range(length): #for each review
            
            cols = np.where(X[i,:]==1)[0]
            P_review_1 = self.P_train_reviews_1*np.prod(self.P_word_1[cols])
            P_review_0 = self.P_train_reviews_0*np.prod(self.P_word_0[cols])

            #Normalization
            temp_1 = P_review_1
            P_review_1 = P_review_1 / (P_review_1 + P_review_0)
            P_review_0 = P_review_0 / (temp_1 + P_review_0)

            Y[i] = 1 if P_review_1 > P_review_0 else 0

        return Y
