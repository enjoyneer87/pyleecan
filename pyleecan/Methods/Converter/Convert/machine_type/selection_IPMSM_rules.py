def selection_IPMSM_rules(self):
    """selection step to have rules for motor IPMSM

    Parameters
    ----------
    self : ConvertMC
        A ConvertMC object

    """
    # step for stator
    self.selection_LamSlotWind_rules()

    is_stator = False
    self.selection_slot_rotor_rules(is_stator)
    self.selection_bar_rules(is_stator)
    self.selection_lamination_rules(is_stator)
    self.selection_skew_rules(is_stator)
