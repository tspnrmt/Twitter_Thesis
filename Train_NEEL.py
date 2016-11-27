from twython import Twython, TwythonError

import csv

print('Hello')
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
    lines_with_texts = []
    for line in file:
        # The rstrip method gets rid of the "\n" at the end of each line
        lines.append(line.rstrip().split('\t'))
        #print(lines[index][0])
        try:
            tweet = twitter.show_status(id=lines[index][0])
            #print(tweet['text'])
            lines_with_texts.append( tweet['text'] +  '\t' +  tweet['text'][int(lines[index][1]):int(lines[index][2])] +  '\t' +
                                     lines[index][5])
            #tweet['text'] [int(lines[index][1]):int(lines[index][2]])
            index = index + 1
        except (TwythonError, AttributeError):
            #print("Not valid ID")
            index = index + 1
        if(index%50 == 0):
            print(index)
        #if(index == 5):
        #    break

    thefile = open('test.txt', 'w')
    for item in lines_with_texts:
        thefile.write("%s\n" % item)



