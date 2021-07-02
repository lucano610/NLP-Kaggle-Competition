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
* have the json file and the csv files are in the same directory as vocab2.py
* run the vocab2.py script
```
python3 vocab2.py
```
The result should be 6 numpy files (.npy files) 2 are for the training set, 2 are for the testing set, 1 is the training set labels, and 1 is the testing set labels. These numpy files will be loaded later on to be used for the model next.

The image below is  visual of how the tokenization of the sample sentences take place using the vocabulary set created from truncator.py

![Tokenization](https://github.com/huda-irs/cs542_NLP_Project/blob/main/images/tokenization.PNG)
## Model

## Results


