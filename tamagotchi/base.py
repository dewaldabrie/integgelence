import os
import numpy as np
from math import log
import pickle
import time
from tamagotchi.settings import LOOP_PERIOD, TAMAGOTCHI_SAVE_DIR, STATUS_ENCODER_DECODER, STATUS_SENDER
from collections import namedtuple, OrderedDict
from typing import Dict, List, Union, Any

# input affects Tamagotchi's internal state
ALLOWABLE_INPUTS = {
            'feed',
            'pet',
            'excercise',
            'immunize',
            'clean',
        }

# interactions affect Tamagotchi's active state
ALLOWABLE_INTERACTIONS = {
            'put to bed', # TODO
            'water',
            'petting',
            'excercise',
            'immunization shot', # TODO
            'shelter', # TODO
        }

class TamagotchiMeta(type):
    pass


class Tamagotchi():
    """
    Define the base attributes and methods of a Tamagotchi.
    """
    # general attributes
    species_name = 'Tamagotchi'
    unique_name = None


    # possible active states
    ACTIVE_STATES = {
        'sleeping',
        'inactive',
        'playing',
        'eating',
        'pooping'
    }

    AGE_STATES = {
        'infant',
        'todler',
        'teenager',
        'adult',
        'aged'
    }

    # physical attributes
    adult_mean_weight = 100  # kg
    weight = adult_mean_weight
    age = 1  # seconds
    _last_update_time = None

    # parameters
    #  takes on values between 0 and 1.0 (inclusive)
    #  mental
    social_affinity = 0.5
    touch_affinity = 0.5
    excercise_affinity = 0.5
    food_affinity = 1.0  # always thinks it's hungry

    #  physical
    @property
    def appetite(self):
        """
        Appetite drops with age
        See notebooks/InteractionModel.ipynb
        """
        return 0.5 + np.log(self.age / 24 / 3600) / log(1. / 24 / 3600) * 0.5

    disease_succeptibility = 0.5
    weather_resistance = 0.5
    excercise_required = 0.5


    # state
    # perfect/healthy state variable has value of zero
    nurishment = -0.5  # hungry <--> overfed
    social = -0.5  # lonely <--> annoid
    fitness = -0.5  # unfit <--> chronic exhaustion
    undiseased = 0  # terminally diseased <--> auto-immune disorder
    state_dict = {
        'nurishment': nurishment,
        'social': social,
        'fitness': fitness,
        'undiseased': undiseased,
    }

    # Create the transformation matrix T to map inputs to state
    #         Input vector : (n,1)
    #         State vector: (m,1)
    #         n is stricly larger or equal to m
    #         Transformation matrix:(m,n)x(n,1) => (m,1)
    #         Transformation matrix such that Snext = Sprev + Sprev (dot) T x I
    #

    # (4,m) x (m,n) x (nx1)
    #  B       A       I

    # Attribute to state mapping
    # 4xm
    # rows have to sum to 1
    #   food_affinity  touch_affinity  social_affinity  excercise_affinity  appetite  excercise_required  disease_succeptibility
    B = np.matrix([
        [0.5,          0.0,             0.0,             0.0,                0.5,     0.0,                 0.0,               ],  # nourishment
        [0.0,          0.5,             0.5,             0.0,                0.0,     0.0,                 0.0,               ],  # social
        [0.0,          0.0,             0.0,             0.5,                0.0,     0.5,                 0.0,               ],  # fitness
        [0.0,          0.0,             0.1,             0.2,                0.2,     0.0,                 0.5,               ],  # undiseased
    ])

    # Input to attribute mapping
    # mxn
    # rows have to sum to 1
    #   'feed', 'pet', 'excercise', 'immunize', 'clean'
    A = np.matrix([
        [1.0,    0.0,    0.0,          0.0,        0.0],    # food_affinity
        [0.0,    1.0,    0.0,          0.0,        0.0],    # touch_affinity
        [0.0,    0.5,    0.5,          0.0,        0.0],    # social_affinity
        [0.0,    0.0,    1.0,          0.0,        0.0],   # excercise_affinity
        [1.0,    0.0,    0.0,          0.0,        0.0],   # appetite
        [0.0,    0.0,    1.0,          0.0,        0.0],   # excercise_required
        [0.0,    0.0,    0.0,          0.1,        0.9],   # disease_succeptibility
    ])

    # health state vector
    state = np.array([
        nurishment,
        social,
        fitness,
        undiseased,
    ])

    # health properties
    # this represents relative importance of the various state variables
    # to the type of health in question
    # these must correspond to the state variables in the same order
    # (nx1) dot (nx1) => (1x1)
    LinAttrComb = namedtuple('LinearAttributeCombination', ('attributes', 'coefficients'))

    mental_health_map = OrderedDict(
        nourishment=LinAttrComb(['food_affinity'], []),
        social=LinAttrComb(['touch_affinity', 'social_affinity'], []),
        fitness=LinAttrComb(['excercise_affinity'], []),
        undiseased=.5,
    )
    mental_health_weights = np.array([
        food_affinity, # nourishment
        (touch_affinity + social_affinity)/2., # social
        excercise_affinity, # fitness
        0.5, # undiseased
    ])

    physical_health_map = OrderedDict(
        nourishment=LinAttrComb(['appetite'], []),
        social=0.,
        fitness=LinAttrComb(['excercise_required'], []),
        undiseased=LinAttrComb(['disease_succeptibility'], []),
    )
    physical_health_weights = np.array([
        appetite, # average size needs average quantity of food
        0.0,
        excercise_required,
        disease_succeptibility,
    ])


    def __init__(self,  unique_name):
        # TODO: enforce uniqueness
        self.unique_name = unique_name
        # update health weights
        self._update_mental_health_weights()
        self._update_physical_health_weights()
        self.encoder_decoder = STATUS_ENCODER_DECODER()

        # configure available publications
        self.status_publisher = STATUS_SENDER
        STATUS_PUB = self.unique_name + '_status'
        self.publicatoins = {
            STATUS_PUB : self.status_publisher(STATUS_PUB),
        }

    @property
    def mental_health(self):
        return self._calc_health(self.mental_health_weights)

    @property
    def physical_health(self):
        return self._calc_health(self.physical_health_weights)

    def _calc_health(self, weights):
        """
        Calculate heath based on the relevant weights and the internal state

        Perfect state variables are zero, but for the sake of human understanding,
        perfect health is represented as unity.

        Health = 1 - normalized_weights (dot) abs(state)

        The abs val of the state is taken because both extremes are bad, for
        example over-eating and under-eating
        """
        # normalize the selection weights
        norm_weights = self._normalize(weights)
        # calculate health
        health = 1.0 - np.dot(
            norm_weights,
            abs(self.state)
        )
        return health

    # TODO: energy properties

    @staticmethod
    def _normalize(vector:np.array) -> np.array:
        """
        Cause non-zero weight vectors to sum to 1.
        :param vector: numpy array with relative weights that indicate relative importance of corresponding values in
        another array that the selection vector will be dotted with
        :return: array that sums to 1, except zero array passes through
        """
        # make sure all values are positive
        if np.min(vector) < 0.:
            vector += vector.min()

        s = vector.sum()
        return vector/s if s > 0 else vector

    def _receive_input(self, input:dict) -> None:
        """
        Recieve an input from another agent and update the
        state (S) as required.

        Snext += B x A x I
        """
        input_mat = self._parse_input(input)

        self.state = self.state + np.squeeze(self.B*self.A*input_mat)
        self._apply_state_limits()

    def _apply_state_limits(self):
        for idx, val in enumerate(self.state):
            self.state[idx] = np.clip(val, -1, 1)


    def _parse_input(self, input_dict:dict) -> np.matrix:
        """
        Parse the input dictionary into an array with dimensions same as that
        of ALLOWABLE_INPUTS

        :return: matrix of dimension nx1 (n being max number of inputs)
        """
        result_dict = dict((name, 0.) for name in ALLOWABLE_INPUTS)
        result_dict.update(input_dict)
        return np.matrix(list(result_dict.values())).T

    def _publish_status(self,
                       topic:str,   # topic of the broadcast
                       message:Any  # encoded message
        ):
        """publish the public state to the outside world"""
        encoded_state = self.encoder_decoder.encode(message)
        # pass message to publiser
        self.publicatoins[topic].send_message(encoded_state)

    def update(self, input=None):
        """
        Update everything leading up to status
        """
        if input:
            self._receive_input(input)

        self._update_time_dependant_vars()

        # pack health status
        health_status = dict(
            mental_health=self.mental_health,
            physical_health=self.physical_health
        )

        self._publish_status(self.unique_name + '_status', health_status)

        return (self.mental_health, self.physical_health)

    def _update_time_dependant_vars(self):
        """
        Update the state variables that depend on time
        """
        # register update time
        if self._last_update_time is None:
            self._last_update_time = time.time() - LOOP_PERIOD
        this_update_time = time.time()

        # update age
        self.age += time.time() - self._last_update_time

        # update health weights
        self._update_mental_health_weights()
        self._update_physical_health_weights()

        # update activity stage / sleep triggers
        self._last_update_time = this_update_time

    def _calc_health_weights(self, vector_constituency:Union[float, int, Dict[str, LinAttrComb]]) -> np.array:
        """
        Update the vector according to the consituency definition.
        """
        updated = []
        for name, lin_comb in vector_constituency.items():
            if type(lin_comb) in [int, float]:
                updated.append(lin_comb)
            elif isinstance(lin_comb, self.LinAttrComb):
                coefficients = lin_comb.coefficients
                if not lin_comb.coefficients:
                    l = len(lin_comb.attributes)
                    coefficients = [1./l]*l  # equal weighting
                value = 0
                for attr_name, weight in zip(lin_comb.attributes, coefficients):
                    value += getattr(self, attr_name) * weight
                updated.append(value)
            else:
                raise TypeError(f'Unexpected type {type(vector_constituency)} of vector_consituency.')

        return np.array(updated)

    def _update_mental_health_weights(self):
        self.mental_health_weights = self._calc_health_weights(self.mental_health_map)

    def _update_physical_health_weights(self):
        self.physical_health_weights = self._calc_health_weights(self.physical_health_map)


    # def __del__(self):
    #     """Serialize self to file"""
    #     with open(os.path.join(TAMAGOTCHI_SAVE_DIR, self.unique_name + '.tamag'), 'w') as f:
    #         pickle.dump(self, f)


