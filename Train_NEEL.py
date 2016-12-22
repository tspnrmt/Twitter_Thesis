from twython import Twython, TwythonError
import time

import csv

print('Hello ' + time.ctime())


CONSUMER_KEY = "nQNq8qkBd538vOhNehrTiB5bz"
CONSUMER_SECRET = "NLK5iq6e5vKxt5MLc3KaGVRz0NhS7yu3oKPgsVRZj19y32rEJF"
OAUTH_TOKEN = "350186026-mQH1ORAmQck4Aqrltk1eN5iTbLoF1MlEzmtFpPjL"
OAUTH_TOKEN_SECRET = "nvvm4lN6akWMqbko54T9dRcIsb5vc1Q6lWCMKinBtI4g8"
twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
twitter = Twython(app_key=CONSUMER_KEY,
        app_secret=CONSUMER_SECRET,
        oauth_token=OAUTH_TOKEN,
        oauth_token_secret=OAUTH_TOKEN_SECRET)
twitter.verify_credentials()

#tweet = twitter.show_status(id=674869443671941120)
#print(tweet['text'])
#print(tweet['text'][93:101])


with open("microposts2016-neel-test_neel.gs") as file:
    lines = []
    index = 0
    index_error = 0
    lines_with_texts = []


    for line in file:
        # The rstrip method gets rid of the "\n" at the end of each line
        lines.append(line.rstrip().split('\t'))
        #print(lines[index][0] )
        try:
            tweet = twitter.show_status(id=lines[index][0])
            #print(tweet['text'])
            temp_NER = ''
            if (lines[index][5] == 'Person'):
                temp_NER = '1'
            elif (lines[index][5] == 'Thing'):
                temp_NER = '2'
            elif (lines[index][5] == 'Organization'):
                temp_NER = '3'
            elif (lines[index][5] == 'Location'):
                temp_NER = '4'
            elif (lines[index][5] == 'Product'):
                temp_NER = '5'
            elif (lines[index][5] == 'Event'):
                temp_NER = '6'
            elif (lines[index][5] == 'Character'):
                temp_NER = '7'
            temp_NER = lines[index][5]
            #else:
            #    temp_NER = lines[index][5]

            #lines_with_texts.append( tweet['text'] +  '\t' +  tweet['text'][int(lines[index][1]):int(lines[index][2])] +  '\t' +
            #                         temp_NER)
            print(tweet)
            lines_with_texts.append(
                lines[index][0] + '\t' + tweet['text'] + '\t' + lines[index][1] + '\t' + lines[index][2] + '\t' +
                temp_NER + '\t' +  str(tweet['retweet_count'])  + '\t' +  str(tweet['favorite_count']) + '\t' +
                tweet['text'] [int(lines[index][1]):int(lines[index][2])])
            #tweet['text'] [int(lines[index][1]):int(lines[index][2]])
            index = index + 1
        except TwythonError as e:
            #print("Not valid ID")
            print (e)
            index = index + 1
            index_error = index_error + 1
        except  AttributeError as a:
            print (a)
            index = index + 1
            index_error = index_error + 1
        if(index%50 == 0):
            print(index)
            print(index_error)
        if ( index != 0 and index%900 == 0):
            print('Sleep --> ' + time.ctime())
            time.sleep(16*60)
            twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
            twitter = Twython(app_key=CONSUMER_KEY,
                              app_secret=CONSUMER_SECRET,
                              oauth_token=OAUTH_TOKEN,
                              oauth_token_secret=OAUTH_TOKEN_SECRET)
            twitter.verify_credentials()
            print('<-- Sleep '+ time.ctime())
        #if(index == 125):
        #    break

    thefile = open('microposts2016-neel-test_neel4.txt', 'w')
    for item in lines_with_texts:
        thefile.write("%s\n" % item)




