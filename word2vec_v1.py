import gensim
import numpy as np

from scipy import spatial

if __name__ == "__main__":
    """
    model_path = "./word2vec_twitter_model.bin"
    print("Loading the model, this can take some time...")
    model = gensim.models.Word2Vec.load_word2vec_format(model_path, binary=True,unicode_errors='ignore')
    print("The vocabulary size is: "+str(len(model.vocab)))
    x = model.most_similar(positive=['woman', 'king'], negative=['man'])
    print(x)
    #model.doesnt_match("breakfast cereal dinner lunch".split())
    #model.similarity('woman', 'man')
    x = model['computer']
    print('COMPUTER')
    print(type(x))
    print(x)
    y = model['xyztmete']
    print('xyztmete')
    print(type(y))
    print(y)
    """
    sentences = [['first', 'king','woman'], ['man', 'queen'], ['breakfast', 'cereal','dinner', 'lunch']]
    model = gensim.models.Word2Vec(size=2, min_count=1)
    model.build_vocab(sentences)
    record = []
    print('type(record): ')
    print(type(record))
    record.append(model['man'])
    record.append(model['woman'])
    print('model[man]')
    print(model['man'])
    print('model[woman]')
    print(model['woman'])
    list = [['first', 'women', 'woman'] ]
    inferred_vector = model.infer_vector(list)
    print('inferred_vector[women]')
    print(inferred_vector['women'])

    print('type(record): ')
    print(type(record))
    record = np.array(record)
    print('type(record): ')
    print(type(record))
    print(record)

    print('average')
    aver = np.average(record, axis=0)
    print(aver)
    result = 1 - spatial.distance.cosine(aver, model['cereal'])
    print('result')
    print(result)
    #x = model.most_similar(positive=['first', 'king'], negative=['man'], topn=1)
    #print(x)
    #y = model.doesnt_match("breakfast cereal dinner lunch".split())
    #print(y)
    #z = model.similarity('woman', 'man')
    #print(z)
    #x = model['man']
    #print('man')
    #print(type(x))
    #print(x)
    #z = model.most_similar("queen")
    #print(z)



