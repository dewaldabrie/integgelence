from .settings import INPUT_SENDER
from settings import PET_PORT_MAP
from tamagotchi.settings import INPUT_ENCODER_DECODER
from tamagotchi.base import ALLOWABLE_INPUTS
from .settings import STATUS_ENCODER_DECODER, STATUS_RECEIVER, LOOP_PERIOD

class Portal:
    """
    Portal into the tamagotchi world.
    Allows the user/owner to interact with his/her virtual pet.
    """
    def __init__(
            self,
            pet_name=None,  # unique name of pet
    ):
        self.pet_name = pet_name
        if not pet_name:
            raise ValueError('Must supply pet_name field, i.e. specify recipient.')
        port = PET_PORT_MAP[pet_name]
        # configure input link to tamagotchi
        self.sender = INPUT_SENDER(
            encoder_decoder=INPUT_ENCODER_DECODER,
            port=port,
        )

        # get messages from pet
        PET_STATUS = pet_name + '_status'  # topic name
        self.subscriptions = {
            'status': STATUS_RECEIVER(PET_STATUS, encoder_decoder=STATUS_ENCODER_DECODER)
        }

    def show_menu(self):
        """show the menu options"""
        options = '\t'.join([f"{idx+1}. {action}" for (idx, action) in enumerate(ALLOWABLE_INPUTS)])
        options += f"\t {len(ALLOWABLE_INPUTS)+1}. Check on Tamagotchi"
        print(options)

    def get_user_input(self):
        """Get and validate user input"""
        i = input().strip()
        if i not in [str(a) for a in range(1,len(ALLOWABLE_INPUTS)+2)]:
            print(f"Invalid input {i}, please try again.\n")
            return
        # show status
        status = self.subscriptions['status'].receive_message()
        print("New status from %s: %s" % (self.pet_name, status))

        # return input in dictionary format, assume unity quantity
        if i != str(len(ALLOWABLE_INPUTS) + 1):
            return {ALLOWABLE_INPUTS[int(i)-1]: 1.0}

    def main(self):
        """Show use the menu and act on his/her input"""
        print("Choose the number of the action you would like to peform for you pet:")
        self.show_menu()
        input = self.get_user_input()
        if input:
            self.sender.send_message(input)
