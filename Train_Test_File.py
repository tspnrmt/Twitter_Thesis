import pandas as pd

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

def expFile(filename):
    with open(filename) as file:
        lines = []
        index = 0
        lines_with_texts = []
        set_tweets = set()
        for line in file:
            # The rstrip method gets rid of the "\n" at the end of each line
            lines.append(line.rstrip().split('\t'))

            index = index + 1
            if (index == 5):
                break
        dict = {}
        for line in lines:
            dict[line[1]] = line[2]
            set_tweets.add(line[0])
        print(dict)
        words = []
        words.append('& Start Doc &')
        for letter in set_tweets:
            words.append('& Start Tweet &')
            for word in letter.split():
                words.append(word)
            words.append('& End Tweet &')
        words.append('& End Doc &')
        print (words)
        data = {
            '1 WB Letter': [''],
            #'1 WB Is Capital': [''],
            # '1 WB Is All Capital': [''],
            # '1 WB Cons Vow Ratio': [''],

            '2 W Letter': [''],
            #'2 W Is Capital': [''],
            # '2 W Is All Capital': [''],
            # '2 W Cons Vow Ratio': [''],

            '3 WA Letter': [''],
            #'3 WA Is Capital': [''],
            # '3 WA Is All Capital': [''],
            # '3 WA Cons Vow Ratio': ['']
        }

        df = pd.DataFrame(data, index=words)

        print (df.index.size)
        for j in range(df.index.size):

            """ Word Before """
            wordbefore=df.index[j - 1]
            if  (df.index[j - 1] == '& End Doc &' or df.index[j - 1] == '& Start Doc &' or df.index[j - 1] == '& Start Tweet &' or df.index[j - 1] == '& End Tweet &' or df.index[j - 1] == '& End Doc &' or df.index[j - 1] is None):
                df.iloc[j]['1 WB Letter'] = 0
                #df.iloc[j]['1 WB Is Capital'] = 0
                # df.iloc[j]['1 WB Is All Capital'] = 0
                # df.iloc[j]['1 WB Cons Vow Ratio'] = 0
            else:
                print('Word before: ')
                print(df.index[j - 1])
                df.iloc[j]['1 WB Letter'] = len(df.index[j - 1])
                #df.iloc[j]['1 WB Is Capital'] = df.index[j - 1].istitle()
                # df.iloc[j]['1 WB Is All Capital'] = df.index[j - 1].isupper()
                # if (countVowels(df.index[j - 1]) > 0):
                #     df.iloc[j]['1 WB Cons Vow Ratio'] = countCons(df.index[j - 1]) / countVowels(df.index[j - 1])
                # else:
                #    df.iloc[j]['1 WB Cons Vow Ratio'] = 0

            """ Word """
            word_ = df.index[j]
            if (df.index[j] == '& Start Doc &' or df.index[j] == '& Start Tweet &' or df.index[j] == '& End Tweet &' or df.index[j] == '& End Doc &' or df.index[j] is None):
                df.iloc[j]['2 W Letter'] = 0
                #df.iloc[j]['2W Is Capital'] = 0
                # df.iloc[j]['2 W Is All Capital'] = 0
                # df.iloc[j]['2 W Cons Vow Ratio'] = 0
            else:
                print('Word: ')
                print(df.index[j])
                df.iloc[j]['2 W Letter'] = len(df.index[j])
                #df.iloc[j]['2 W Is Capital'] = df.index[j].istitle()
                # df.iloc[j]['2 W Is All Capital'] = df.index[j].isupper()
                # if (countVowels(df.index[j]) > 0):
                #     df.iloc[j]['2 W Cons Vow Ratio'] = countCons(df.index[j]) / countVowels(df.index[j])
                # else:
                #     df.iloc[j]['2 W Cons Vow Ratio'] = 0

            """ Word After """
            if (j+2 > df.index.size):
                break
            wordafter = df.index[j + 1]
            if (df.index[j+1] == '& Start Tweet &' or df.index[j+1] == '& End Tweet &' or df.index[j+1] == '& End Doc &' or df.index[j+1] is None or df.index[j] == '& Start Tweet &'):
                df.iloc[j]['3 WA Letter'] = 0
                #df.iloc[j]['3 WA Is Capital'] = 0
                # df.iloc[j]['3 WA Is All Capital'] = 0
                # df.iloc[j]['3 WA Cons Vow Ratio'] = 0
            else:
                print('Word after: ')
                print(df.index[j + 1])
                df.iloc[j]['3 WA Letter'] = len(df.index[j + 1])
                #df.iloc[j]['3 WA Is Capital'] = df.index[j + 1].istitle()
                # df.iloc[j]['3 WA Is All Capital'] = df.index[j + 1].isupper()
                # if (countVowels(df.index[j + 1]) > 0):
                #    df.iloc[j]['3 WA Cons Vow Ratio'] = countCons(df.index[j + 1]) / countVowels(df.index[j + 1])
                # else:
                #    df.iloc[j]['3 WA Cons Vow Ratio'] = 0
        df = df.drop(['& Start Doc &','& Start Tweet &','& End Tweet &','& End Doc &'])
        print(df)
        df = df.astype(int)
        print(df)
        return df

def main():
    df_train = expFile('microposts2016-neel-training_neel1.txt')
    df_test = expFile('microposts2016-neel-test_neel1.txt')


if __name__ == "__main__":
    main()