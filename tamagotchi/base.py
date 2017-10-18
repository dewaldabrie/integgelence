import numpy as np
from math import log
import time
from tamagotchi.settings import LOOP_PERIOD, TAMAGOTCHI_SAVE_DIR, STATUS_ENCODER_DECODER, STATUS_SENDER, INPUT_RECEIVER, \
    INPUT_ENCODER_DECODER
from collections import namedtuple, OrderedDict
from typing import Dict, List, Union, Any
from settings import PET_PORT_MAP
from functools import lru_cache
import zmq

# input affects Tamagotchi's internal state
ALLOWABLE_INPUTS = [
            'clean',
            'excercise',
            'feed',
            'immunize',
            'pet',
        ]


class TamagotchiMeta(type):
    """
    TODO: Implement meta-class for Tamagotchi factory
    """
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
        'infant':'infant',
        'todler':'todler',
        'teenager':'teenager',
        'adult':'adult',
        'aged':'aged',
    }

    # physical attributes
    adult_mean_weight = 100  # kg
    weight = adult_mean_weight
    adult_mean_age = 400

    _last_update_time = time.time()
    @property
    def age(self):
        return self._time_speedup_factor * (time.time() - self._time_born)

    @property
    def life_stage(self):
        age = self.age
        if age < 60:
            return self.AGE_STATES['infant']
        elif age < 120:
            return self.AGE_STATES['todler']
        elif age < 180:
            return self.AGE_STATES['teenager']
        elif age < 240:
            return self.AGE_STATES['adult']
        elif age < 320:
            return self.AGE_STATES['aged']


    # personality/genetic parameters
    #  takes on values between 0 and 1.0 (inclusive)
    #  mental
    social_affinity = 0.5
    touch_affinity = 0.5
    excercise_affinity = 0.5
    food_affinity = 1.0  # always thinks it's hungry

    #  physical
    disease_succeptibility = 0.5
    weather_resistance = 0.5
    excercise_required = 0.5

    @property
    def appetite(self):
        """
        Appetite drops with age
        See notebooks/InteractionModel.ipynb
        """
        return 0.5 + np.log(self.age / 24 / 3600) / log(1. / 24 / 3600) * 0.5


    # state
    STATE_NAMES = [
        'fitness',
        'nourishment',
        'social_stimulation',
        'undiseased',
    ]

    # initial state values: perfect/healthy state variable has value of zero
    nurishment = -0.5  # hungry <--> overfed
    social = -0.5  # lonely <--> annoid
    fitness = -0.5  # unfit <--> chronic exhaustion
    undiseased = 0  # terminally diseased <--> auto-immune disorder

    # Model health degradation with age
    #         State vector (S): (1,m)
    #         Mental/physical age sensitivities (C): (1,m)
    #         Age delta since last update (r): time_passed/nominal_age
    #
    #         Transformation:(1,m) - (1,m).r => (1,m)
    #         Transformation matrix such that S = S_prev - C elemwise r
    #
    C = np.matrix([
        0.8,  # nourishment state sensitvity to age (0 - 1)
        0.1,  # social state sensitvity to age (0 - 1)
        0.6,  # fitness state sensitvity to age (0 - 1)
        0.9,  # undiseased state sensitvity to age (0 - 1)
    ]).T



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
        [0.0,          0.0,             0.0,             0.5,                0.0,     0.5,                 0.0,               ],  # fitness
        [0.5,          0.0,             0.0,             0.0,                0.5,     0.0,                 0.0,               ],  # nourishment
        [0.0,          0.5,             0.5,             0.0,                0.0,     0.0,                 0.0,               ],  # social
        [0.0,          0.0,             0.1,             0.2,                0.2,     0.0,                 0.5,               ],  # undiseased
    ])

    # Input to attribute mapping
    # mxn
    # rows have to sum to 1
    #   'clean',  'excercise',  'feed', 'immunize', 'pet',
    A = np.matrix([
        [0.0,       0.0,         1.0,      0.0,      0.0,    ],    # food_affinity
        [0.0,       0.0,         0.0,      0.0,      1.0,    ],    # touch_affinity
        [0.0,       0.5,         0.0,      0.0,      0.5,    ],    # social_affinity
        [0.0,       1.0,         0.0,      0.0,      0.0,    ],   # excercise_affinity
        [0.0,       0.0,         1.0,      0.0,      0.0,    ],   # appetite
        [0.0,       1.0,         0.0,      0.0,      0.0,    ],   # excercise_required
        [0.9,       0.0,         0.0,      0.1,      0.0,    ],   # disease_succeptibility
    ])


    # health properties
    # this represents relative importance of the various state variables
    # to the type of health in question
    # these must correspond to the state variables in the same order
    # (nx1) dot (nx1) => (1x1)
    LinAttrComb = namedtuple('LinearAttributeCombination', ('attributes', 'coefficients'))

    mental_health_map = OrderedDict(
        fitness=LinAttrComb(['excercise_affinity'], []),
        nourishment=LinAttrComb(['food_affinity'], []),
        social=LinAttrComb(['touch_affinity', 'social_affinity'], []),
        undiseased=.5,
    )

    physical_health_map = OrderedDict(
        fitness=LinAttrComb(['excercise_required'], []),
        nourishment=LinAttrComb(['appetite'], []),
        social=0.,
        undiseased=LinAttrComb(['disease_succeptibility'], []),
    )


    def __init__(self,  unique_name:str, time_speedup_factor:float=1.0):
        # TODO: enforce uniqueness
        self.unique_name = unique_name
        self._time_speedup_factor = time_speedup_factor
        self._time_born = time.time()

        # initialize state
        self.state = np.matrix([
            self.fitness,
            self.nurishment,
            self.social,
            self.undiseased,
        ]).T

        # configure available publications
        STATUS_PUB = self.unique_name + '_status'
        self._publications = {
            STATUS_PUB : STATUS_SENDER(STATUS_PUB, STATUS_ENCODER_DECODER),
        }

        # configure link to owner (for receiving inputs)
        self.input_reader = INPUT_RECEIVER(
            encoder_decoder=INPUT_ENCODER_DECODER,
            port=PET_PORT_MAP[self.unique_name]
        )


    def update(self, input=None):
        """
        Update everything leading up to status. This should be called externally from process the loop.
        """
        self._process_input_queue()
        self._update_time_dependant_vars()

        status = {}

        # pack health status
        health_status = dict(
            mental_health=self.mental_health,
            physical_health=self.physical_health
        )

        # age and age group
        age_status = dict(
            age=self.age,
            life_stage=self.life_stage,
        )

        # convert state array into dict with labels
        state_dict = dict((k,v) for k,v in zip(Tamagotchi.STATE_NAMES, self.state.T.tolist()[0]))

        status.update(state_dict)
        status.update(health_status)
        status.update(age_status)

        # broadcast to the world
        self._publish_status(self.unique_name + '_status', status)

        return (self.mental_health, self.physical_health)

    @property
    def mental_health(self):
        return float(self._calc_health(self._mental_health_weights))

    @property
    def physical_health(self):
        return float(self._calc_health(self._physical_health_weights))

    # @lru_cache(maxsize=None)
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
        # assert self.state.shape == (5,1), "Shape is actually %s" % self.state.shape
        health = np.dot(
            norm_weights,
            self._map_state_to_health_space(self.state)
        )
        return health

    @staticmethod
    def _map_state_to_health_space(state:np.array) -> np.array:
        """
        Model that perfect state is zero, -1 is lack, +1 is overdone.
        See notebook InteractionModel for plot of function
        """
        return -1 * abs(state) + 1

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

    @property
    def _mental_health_weights(self):
        return self._calc_health_weights(self.mental_health_map)

    @property
    def _physical_health_weights(self):
        return self._calc_health_weights(self.physical_health_map)

    # @lru_cache(maxsize=None)
    def _calc_health_weights(self, vector_constituency: Union[float, int, Dict[str, LinAttrComb]]) -> np.array:
        """
        Update the vector according to the consituency definition.
        """
        updated = []
        for name, lin_comb in vector_constituency.items():
            # for hardcoded values that don't use linear combinations
            if type(lin_comb) in [int, float]:
                updated.append(lin_comb)
            # for linear combinations
            elif isinstance(lin_comb, self.LinAttrComb):
                coefficients = lin_comb.coefficients
                if not lin_comb.coefficients:
                    l = len(lin_comb.attributes)
                    coefficients = [1. / l] * l  # equal weighting
                value = 0
                for attr_name, weight in zip(lin_comb.attributes, coefficients):
                    value += getattr(self, attr_name) * weight
                updated.append(value)
            else:
                raise TypeError(f'Unexpected type {type(vector_constituency)} of vector_consituency.')
        return np.array(updated)


    def _process_input_queue(self) -> None:
        """
        Recieve an input from another agent and update the
        state (S) as required.

        Snext += B x A x I
        """
        for message in self.input_reader.receive_messages():
            input_mat = self._parse_input(message)
            self.state = self.state + self.B*self.A*input_mat
            assert(self.state.shape, (4,1))
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
        # keep alphabetical order in keys
        sorted_tuples = sorted(result_dict.items(), key=lambda t:t[0])
        input_matrix =  np.matrix([ v for (k,v) in sorted_tuples]).T
        return input_matrix

    def _publish_status(self,
                        topic:str,   # topic of the broadcast
                        message:Any  # encoded message
                        ):
        # pass message to publiser
        self._publications[topic].send_message(message)

    def _update_time_dependant_vars(self):
        """
        Update the state variables that depend on time.
        The mental and physical state decay with time.

        S = S_prev - C elmwise (time_lapsed/nominal_age)
        """
        # register update time
        if self._last_update_time == 0:
            self._last_update_time = time.time() - LOOP_PERIOD
        this_update_time = time.time()

        # update activity state with age degradation
        self.state -= np.multiply(self.C, (LOOP_PERIOD/self.adult_mean_age))
        self._apply_state_limits()


        self._last_update_time = this_update_time

