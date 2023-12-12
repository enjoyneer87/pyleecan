from .....Classes.CondType12 import CondType12
from .....Classes.CondType13 import CondType13


def select_conductor_rules(self, is_stator):
    """selection step to add rules for conductor

    Parameters
    ----------
    self : ConvertMC
        A ConvertMC object
    is_stator : bool
        True slot is in stator, False slot is in rotor

    """

    # select slot type and add it to obj machine or in dict
    if self.is_P_to_other:
        self.convert_conductor_to_other()
    else:
        self.convert_conductor_to_P()

    # add the correct rule depending on the rotor
    if isinstance(self.machine.stator.winding, CondType12):
        self.add_rule_condtype12()

    elif isinstance(self.machine.stator.winding, CondType13):
        self.add_rule_condtype13()
