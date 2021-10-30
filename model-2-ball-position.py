# infering the position of a ball in a grid

# We have an NxM grid and place a ball in one cell at random.
# Each cell has an equal chance of being picked.
#
# Our goal is to figure out where the ball is most likely to be, given information about whether a
# randomly chosen cell is north, south, east or west of the ball


import numpy as np
from numpy.random import uniform
from time import sleep
from os import system

width = 10
height = 10
number_of_observations = 200


# ball position

position = uniform((width, height)).astype(int)


# generate observations

def information_about_position_relative_to_ball(pos):
    return [pos[0] < position[0], pos[1] < position[1]]


positions = np.array([

    information_about_position_relative_to_ball(uniform((width, height)).astype(int))
    for i in range(number_of_observations)

])


# assign priors
# objectively, each cell should have an equal prior probability because we know they were picked
# uniformly at random

probs = np.ones((width, height)) / (width * height)


# beleif updating method


def update(observation):
    global probs

    left = observation[0]
    above = observation[1]

    for row in range(height):
        for col in range(width):

            # joint distribution p(above, left | <x, y>) = p(above | <x, y>) * p(left | <x, y>)
            # = p(above | y) * p(left | x)

            prow = row / height if above else 1 - (row / height)
            pcol = col / width if left else 1 - (col / width)
            pob = prow * pcol

            probs[row][col] *= pob

    probs = probs / probs.sum()



def main():
    for i in range(number_of_observations):
        sleep(0.2)
        print('\033[H\033[2J') # clear
        print('Infering Position of Ball')
        print(f'observation #{i}\n')
        print(probs.round(2))
        update(positions[i])

    print('true position: ', position)


main()
