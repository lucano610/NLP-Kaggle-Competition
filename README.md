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

## Model

## Results
