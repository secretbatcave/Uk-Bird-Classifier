import json
import time
import Queue
import pickle
import pprint
import random
import threading

import obj
import tweepy
import requests
from PIL import Image
pp = pprint.PrettyPrinter(indent=4)

#pylint: disable=bad-whitespace


class TweetShit(object):
    '''
    A class for twittering shit about birds using ML
    '''

    def __init__(self):
        self.lastMentionId = 0
        self.keys = []
        self.api = None

        try:
            self.lastMentionId = pickle.load(open('state.pickle','rb'))
        except Exception as e:
            print "Couldn't open pickle state file: {0}".format(e)
    def __del__(self):
        pickle.dump(self.lastMentionId, open('state.pickle','wb'))

    def loadKeys(self):
        with open('creds.json') as credentials:
            self.keys = json.load(credentials)
    def downloadImage(self, url, tweetId):
        '''download all the picutres
        '''
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open('images/{0}.jpg'.format(tweetId), 'wb') as f:
                for chunk in r:
                    f.write(chunk)
        else:
            return False
    def sassyComment(self, confidence):
        """For a given confidence return a sassy comment
        """
        if confidence < 60:
            comment = ['It could be a', 'Hmmm, might be a', 'I suspect its a', 'May contain']
        if confidence >= 60 and confidence < 75:
            comment = ['Pretty sure its a', 'I\'m pretty sure its a', 'Thats a', ]
        if confidence >= 75:
            comment = ['I\'m sure its a', 'It is a']
        return random.choice(comment)

    def markReadTweet(self, tweetId):
        '''takes the twitter ID and makes sure that
        we don't see it again
        '''
        if self.lastMentionId < tweetId:
            self.lastMentionId = tweetId

    def login(self):
        keys = self.keys
        auth = tweepy.OAuthHandler(keys['consumerKey'],keys['consumerSecret'])
        auth.set_access_token(keys['accessToken'],keys['AccessTokenSecret'])

        self.api=tweepy.API(auth)
    def getMentions(self):
        '''Returns any new status since last invocation
        tweets are marked as read when this is called.
        '''
        if self.lastMentionId:
            mentions = self.api.mentions_timeline(since_id=self.lastMentionId)
        else:
            mentions = self.api.mentions_timeline()
        if mentions:
            self.markReadTweet(mentions[0].id)

            return reversed(mentions)
        return False
    def sendReply(self, user, imagePath, replyId, imageResults ):
        '''MAkes a nice message and replay with
        images
        '''
        birdType = ''
        confidence = 0
        for result in imageResults:#pylint: disable=redefined-outer-name
            if 'bird species' in result:
                sassyComment = self.sassyComment(round(result['bird species'][0][1]*100))
                confidence = round(result['bird species'][0][1]*100)
                birdType = result['bird species'][0][0]
            else:
                sassyComment = self.sassyComment(round(result['confidence']))
                confidence = round(result['confidence'])
                birdType = result['class']['name']


        message = "@{0} {1} {2} ({3}%) ".format(user, sassyComment, birdType, confidence)
        outputMessage = self.api.update_with_media(imagePath, status = message, in_reply_to_status_id=replyId)
        return outputMessage



if __name__ == '__main__':
    taskQ = Queue.Queue()
    resultQ = Queue.Queue()
    dave = TweetShit()
    dave.loadKeys()
    api = dave.login()
    paul = obj.detectObjects(taskQ, resultQ)
    paul.PATH_TO_BIRD = '../models/ukGardenModel.pb'
    paul.PATH_TO_CKPT = 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
    paul.loadObjectGraph()
    paul.loadClassifyerGraph()
    thread = threading.Thread(target = paul.objectDetectionLoop)
    thread.daemon = True
    thread.start()
    while True:
        tweets = dave.getMentions()
        if not tweets:
            print "no new tweets"
        else:
            for tweet in tweets:
                print tweet.text
                print tweet.user.screen_name
                print tweet.geo
                print tweet.place
                print tweet.id
                if 'media' in tweet.entities:
                    dave.downloadImage(tweet.entities['media'][0]['media_url_https'], tweet.id)
                    pp.pprint (tweet.entities['media'])
                    taskQ.put('images/{0}.jpg'.format(tweet.id))
                    results = resultQ.get(block=True)
                    image = Image.open('images/{0}.jpg'.format(tweet.id))
                    classifiy = False
                    for result in results:
                        print result['location']
                        ymin, xmin, ymax, xmax = result['location']
                        print result['class']['name']
                        print result['confidence']
                        classifiy = True
                        if 'bird species' in result:
                            pp.pprint(result)
                            label = "{0} {1}%".format(result['bird species'][0][0], round(result['bird species'][0][1]*100))
                            obj.draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color='red', thickness=4,display_str_list=( "{0}".format(label), ),use_normalized_coordinates=False)
                        else:
                            obj.draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color='red', thickness=4,display_str_list=( "{0}:{1}%".format(result['class']['name'],result['confidence']), ),use_normalized_coordinates=False)

                        #pp.pprint(result)
                    if results:
                        print "saving annotated image in png"
                        image.save('images/{0}-annotated.png'.format(tweet.id), "PNG")
                        output = dave.sendReply(tweet.user.screen_name, 'images/{0}-annotated.png'.format(tweet.id), tweet.id , results)
                        print output
        time.sleep(120)

    #api.update_status("@{0} good afternoon".format(mentions[0].user.screen_name),  mentions[0].id)
