import numpy as np
from math import log

class Tamagotchi:
    """
    Define the base attributes and methods of a Tamagotchi.
    """
    # general attributes
    species_name = 'Tamagotchi'

    # physical attributes
    adult_mean_weight = 100  # kg
    age = 1./365  # years

    # parameters
    #  takes on values between -1.0 and 1.0 (inclusive)
    #  mental
    social_affinity = 0.5
    touch_affinity = 0.5
    excercise_affinity = 0.5
    theta_mental = np.array([
        social_affinity,
        touch_affinity,
        excercise_affinity,
    ])

    #  physical
    appetite = lambda age: 0.5 + np.log(age) / log(1. / 365) * 0.5  # appetite drops with age
    immune_function = 0.5
    weather_resistance = 0.5
    excercise_required = 0.5
    theta_phys = np.array([
        immune_function,
        weather_resistance,
        excercise_required
    ])

