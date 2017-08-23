# A twitter bot

This prototype twitterbot sits in a loop waiting for people to tweet images to it. 

Once it receives an image it can open it will force it through two tensor flow models. The first model is the stock object detection model from google. This picks out and crops anything that it thinks is a bird. 

It then passes the cropped image to my custom trained classifier. This is to give more accurate results, and classify multiple birds in the same image

## Why are you using queues?

`obj.py` is a bit of a hack. First it takes the tensorflow example and hacks its about, it also is designed to be a long running thread.  This is because loading the two models is very expensive (> 5 seconds) compared with the actual detection and classification (<1sec)   

Therefore two queues are used to feed input paths into the models, and get results out again.
