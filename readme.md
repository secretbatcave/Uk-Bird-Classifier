# UK bird classifier

This is a pre-trained inception v3 tensorflow model that should be able to correctly classify common UK
birds. 

# What can it detect:

Good question, there are two models currently, (look in the models folder.) `largeBirds4.pb` is trained on many many more species of birds, with a slight drop in accuracy, where as `ukGarenModel.pb` is trained on only 12. 

The data I've trained this model on is fairly specific. All animals were closely cropped, this means that to get good results you'll need to make sure that your image you wish to classify is side on and quite big.

The list of animals that `ukGardenBirds.pb` has been trained on are:

1. Gray Squirrels
2. Crow (carrion and rook)
3. Wren
4. Wood pidgeon
5. House sparrow
6. Magpie
7. Blackbird (male and female, although female is less well defined in the model)
8. Chaffinch
9. Dunnock
10. Song Thrush
11. Robin (male and female) 
12. Cats

LargeBirds4 however has many more species it can detect:

1.  pheasant
1.  grey partridge
1.  wren
1.  black restart
1.  grey heron
1.  yellow wagtail
1.  starling
1.  goldfinch
1.  stock dove
1.  jackdaw
1.  squirrel
1.  great tit
1.  jay
1.  long tailed tit
1.  crossbill
1.  collared dove
1.  house sparrow
1.  mistle thrush
1.  chaffinch
1.  wood pigeon
1.  bullfinch
1.  coal tit
1.  pied wagtail
1.  blackheaded gull
1.  meadow pipit
1.  treecreeper
1.  bluetit
1.  lesser whitethroat
1.  turtle dove
1.  grey wagtail
1.  dunnock
1.  reed bunting
1.  swift
1.  ring necked parakeet
1.  green woodpecker
1.  siskin
1.  stonechat
1.  waxwing
1.  rook
1.  chiffchaff
1.  cuckoo
1.  house martin
1.  spotted flycatcher
1.  nuthatch
1.  cat
1.  pied flycatcher
1.  swallow
1.  great black backed gull
1.  blackbird
1.  crested tit
1.  lesser spotted woodpecker
1.  robin
1.  herring gull

This should cover most common birds in your garden. 

# How to run

First you need to make sure that [Tensorflow](https://www.tensorflow.org/) is [installed](https://www.tensorflow.org/install/)

Then once you have cloned this repo, run:

```python bird.py``` 

which will classify any pictures in the "testImages" folder 

The script "bird.py" is just the example from tensor flow, and has no special magic to make it run (apart from the blackbox that is TF...)

# Dataset

I have not included the dataset, as I don't own the copyright. However if you wish to obtain a copy, please get in touch
