#!usr/bin/env python
import letterLangId
import wordLangId
def letter_lang_id_script():
    'letter bigram model for language ID'
    data = letterLangId.get_data()
    model = letterLangId. train(data['train'])
    predicted_values = letterLangId. predict(data['test'], model)
    accuracy = letterLangId. evaluate(predicted_values)
    letterLangId. write_solution(predicted_values)

def word_lang_id_script():
    'word bigram model for language ID'
    data = wordLangId.get_data()
    model = wordLangId.train(data['train'])
    predicted_values = wordLangId.predict(data['test'], model)
    accuracy = letterLangId.evaluate(predicted_values)
    wordLangId.write_solution(predicted_values)

if __name__ == '__main__':
    print "This is part I: language identification by letter\n"
    letter_lang_id_script()

    print "This is part II: language identification by word\n"
    word_lang_id_script()
