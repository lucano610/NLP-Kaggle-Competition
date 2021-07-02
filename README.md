# cs542_NLP_Project

## Preprocessing
  #### Running code 
  The preprocessing should be included. If not **vocab2.py** handles most of the preprocessing.


### Sanitization

### Vocabulary
To create the vocabulary, the you need to create a way to tell which words hold the most valuable data to the problem. We decided that the words that were used the least were less valuable to solving the problem. To find a cut off for what was the least valuable. We looked at the the words sorted words into groups of based on how often they were used. We took the hueristic approach that the lergest group of the least used words should be removed. The cut off we found was around 10. The stop, start, unknown, and padding tokens were added. And the list of words was enumerated.
#### Running code 
The vocabulary **"/trunc/vocabulary_dic.json"** should already be included in the folder, as long as the training data hasn't changed. If not, **"/trunc/truncater.py"** must be run first.

### Tokenization
The Tokenization processing includes standarizing the setences, spilting the samples of setences into a sublevel (a word in this project) which will be the tokens, vectorizing the samples based on the vobaulary created (as described above), and formating all saamples to be the same length in the number of tokens. To implement the tokeninzation the following steps must be followed.
* download the traing, test, and sample csv files from the kaggle compeition [here](https://www.kaggle.com/c/contradictory-my-dear-watson/data)
* have the json file and the csv files from step 1 in the same directory as vocab2.py
* run the vocab2.py script
```
python3 vocab2.py
```

## Model

## Results


