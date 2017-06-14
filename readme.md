# UK bird classifier

This is a pre-trained inception v3 tensorflow model that should be able to correctly classify common UK
birds. 

# What can it detect:

So, the data I've trained this model on is fairly specific. All animals were closely cropped, this means that to get good results you'll need to make sure that your image you wish to classify is side on and quite big.

The list of animals it has been trained on are:

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

This should cover most common animals in your garden. 

# How to run

First you need to make sure that [Tensorflow](https://www.tensorflow.org/) is [installed](https://www.tensorflow.org/install/)

Then once you have cloned this repo, run:

```python bird.py``` 

which will classify any pictures in the "testImages" folder 

The script "bird.py" is just the example from tensor flow, and has no special magic to make it run (apart from the blackbox that is TF...)

