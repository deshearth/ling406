++++++++++++++++++++++++
+  THIS IS IMPORTANT   +
++++++++++++++++++++++++

The functions are in letterLangId.py and wordLangId.py
and LangIdScript.py is the main function
They are all in <work directory>/bin

In order to execute the program, put training data in 
<word directory>/database; put test data in <work directory>/input
and put solution in <word direcotry>/solution and go to bin folder
run ./LangIdScript.py in terminal
All outputs would be in <work directory>/output

some requirement to run the program:
environment python 2.x
python library: numpy, pandas

-----------------------------------------
Part 1
ans:
Out of 300 test cases, the model gets 299 sentences right.

I think bigram model has to use smooth function.
There are two main reasons.
First, if the single char is 0 in training set. The conditional
probability would be undefined when doing the division.
Second, if the bichar is 0 in certain language A and single char
exists is non-zeros, then when we try to get the the join 
probability of test sentence. One of the conditional probability
would be 0, which makes the join probability would be 0 as well, 
which is not necessarily the case. 

Part 2
ans:
------------------------------------------------------
Models |     letter bigram     |     word bigram      |
------------------------------------------------------
pros   | shorter running time  |    higher accuracy   |
------------------------------------------------------
cons   |      lower accuray    |  longer running time |
------------------------------------------------------
Analysis:
If the measure is the accuracy with the tolerable running time,
I would say in this task, letter bigram is better than the word
bigram model given both of them use laplace smoothing method.
The reason why the letter bigram performs better than word bigram
model is as follows:
The assumption that why bigram model works is that the sequence of
the word or alphabet in each language has some underlying distribution.
The training data can be considered as the sampling from the collections. We
can use the distributions of the samples to approximate the true distributions.
Therefore, if the training data is unbaised, we can get a good approximation.
In other words, the training set has to be as representative as possible.
Since the length of word varies and there are also more than one arrangement
given a set of alphabets, the ways how the words can be formed is a lot, let
alone the ways of one certain word is followed by another certain word.
Therefore, there would be a lot of bi-word sequence in the testing sentences
that would not exist in the training data.
In comparison, the number of ways of combining a bi-char is a lot smaller. In
this way, it is easier for the training data to cover as many cases as possible
given the same corporus.
As for the fact that letter bigram model runs more slowly than word bigram
model, it is obviously because that the for we need to count more chars than
words.


