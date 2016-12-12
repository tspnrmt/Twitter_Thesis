import pandas as pd
import numpy as np

def countVowels(string):
    vowel=("aıioöuüAEIİOÖUÜ")
    count=0
    for i in string:
        if i in vowel:
            count +=1
    return count

def countCons(string):
    cons=("bcçdfgğhjklmnprsştyzBCÇDFGĞHJKLMNPRSŞTVYZ")
    count=0
    for i in string:
        if i in cons:
            count +=1
    return count

def main():
    data = {
            'Word Before': [''],
            'WB Stem': [''],
            'WB POS Tag': [''],
            'WB Letter': [''],
            'WB Letter Diff Stem': [''],
            'WB Is Capital': [''],
            'WB Is All Capital': [''],
            'WB Has Punct BA': [''],
            'WB Has Emot BA': [''],
            'WB Has Double Consonant': [''],
            'WB Has Double Vowel': [''],
            'WB Has Harmony': [''],
            'WB Cons Vow Ratio': [''],
            
            'Word': [''],
            'W Stem': [''],
            'W POS Tag': [''],
            'W Letter': [''],
            'W Letter Diff Stem': [''],
            'W Is Capital': [''],
            'W Is All Capital': [''],
            'W Has Punct BA': [''],
            'W Has Emot BA': [''],
            'W Has Double Consonant': [''],
            'W Has Double Vowel': [''],
            'W Has Harmony': [''],
            'W Cons Vow Ratio': [''],

            'Word After': [''],
            'WA Stem': [''],
            'WA POS Tag': [''],
            'WA Letter': [''],
            'WA Letter Diff Stem': [''],
            'WA Is Capital': [''],
            'WA Is All Capital': [''],
            'WA Has Punct BA': [''],
            'WA Has Emot BA': [''],
            'WA Has Double Consonant': [''],
            'WA Has Double Vowel': [''],
            'WA Has Harmony': [''],
            'WA Cons Vow Ratio': ['']
    }

    words = []
    with open('1.txt','r') as f:
        for line in f:
            for word in line.split():
                words.append(word)


    df = pd.DataFrame(data, index = words)


    print ('.....')
    """for i, row in df.iterrows():
        row['Word'] = i
        row['W Stem'] = i
        row['W POS Tag'] = 'X'
        row['W Letter'] = len(i)
        row['W Letter Diff Stem'] = len(i) - len(i)
        row['W Is Capital'] = i.istitle()
        row['W Is All Capital'] = i.isupper()

    print (df)
    """

    print ('XXXX XXXX')


    for j in range(df.size):
        
        """ Word Before """
        df.iloc[j]['Word Before'] = df.index[j-1]
        df.iloc[j]['WB Stem'] = df.index[j-1]
        df.iloc[j]['WB POS Tag'] = 'X'
        df.iloc[j]['WB Letter'] = len(df.index[j-1])
        df.iloc[j]['WB Letter Diff Stem'] = len(df.index[j-1]) - len(df.index[j-1])
        df.iloc[j]['WB Is Capital'] = df.index[j-1].istitle()
        df.iloc[j]['WB Is All Capital'] = df.index[j-1].isupper()
        df.iloc[j]['WB Has Punct BA'] = ''
        df.iloc[j]['WB Has Emot BA'] = ''
        df.iloc[j]['WB Has Double Consonant'] = ''
        df.iloc[j]['WB Has Double Vowel'] = ''
        df.iloc[j]['WB Has Harmony'] = ''
        if ( countVowels(df.index[j-1]) > 0 ):
            df.iloc[j]['WB Cons Vow Ratio'] = countCons(df.index[j-1]) / countVowels(df.index[j-1])
        else:
            df.iloc[j]['WB Cons Vow Ratio'] = 0
        
        """ Word """
        df.iloc[j]['Word'] = df.index[j]
        df.iloc[j]['W Stem'] = df.index[j]
        df.iloc[j]['W POS Tag'] = 'X'
        df.iloc[j]['W Letter'] = len(df.index[j])
        df.iloc[j]['W Letter Diff Stem'] = len(df.index[j]) - len(df.index[j])
        df.iloc[j]['W Is Capital'] = df.index[j].istitle()
        df.iloc[j]['W Is All Capital'] = df.index[j].isupper()
        df.iloc[j]['W Has Punct BA'] = ''
        df.iloc[j]['W Has Emot BA'] = ''
        df.iloc[j]['W Has Double Consonant'] = ''
        df.iloc[j]['W Has Double Vowel'] = ''
        df.iloc[j]['W Has Harmony'] = ''
        if ( countVowels(df.index[j]) > 0 ):
            df.iloc[j]['W Cons Vow Ratio'] = countCons(df.index[j]) / countVowels(df.index[j])
        else:
            df.iloc[j]['W Cons Vow Ratio'] = 0
        
        """ Word After """
        df.iloc[j]['Word After'] = df.index[j+1]
        df.iloc[j]['WA Stem'] = df.index[j+1]
        df.iloc[j]['WA POS Tag'] = 'X'
        df.iloc[j]['WA Letter'] = len(df.index[j+1])
        df.iloc[j]['WA Letter Diff Stem'] = len(df.index[j+1]) - len(df.index[j+1])
        df.iloc[j]['WA Is Capital'] = df.index[j+1].istitle()
        df.iloc[j]['WA Is All Capital'] = df.index[j+1].isupper()
        df.iloc[j]['WA Has Punct BA'] = ''
        df.iloc[j]['WA Has Emot BA'] = ''
        df.iloc[j]['WA Has Double Consonant'] = ''
        df.iloc[j]['WA Has Double Vowel'] = ''
        df.iloc[j]['WA Has Harmony'] = ''
        if ( countVowels(df.index[j+1]) > 0 ):
            df.iloc[j]['WA Cons Vow Ratio'] = countCons(df.index[j+1]) / countVowels(df.index[j+1])
        else:
            df.iloc[j]['WA Cons Vow Ratio'] = 0
        
        if (j==370):
            break
    print (df)


    print ('.....')

main()