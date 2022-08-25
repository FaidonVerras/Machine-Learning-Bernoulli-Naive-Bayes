# Machine Learning using Bernouli Naive Bayes 

This is a 5th semester AUEB Artificial Inteligence project on Machine Learning.

The goal was to write an algorithm that decides if a review is positive or negative from the "Large Movie Review Dataset" also known as "IMDB Dataset" .
<br><br>

## Machine Learning Libraries used:

- **TensorFlow** to import the Dataset<br>
- **Scikit-learn** to split the data into train and test sets<br><br>

## Comparison with Naive Bayes from Scikit-learn
On average for a 50/50 train-test split dataset including the 1000 most used words.

|           | My NB | Sklearn NB |
|-----------| ----- | ---------- |
| accuracy  | 0.82  | 0.83       |
| precision | 0.83  | 0.83       |
| recall    | 0.80  | 0.83       |
| f1        | 0.81  | 0.83       |