from pyleecan.Classes.LamSlotWind import LamSlotWind
from pyleecan.Classes.SlotW11 import SlotW11


def convert_slot_type(self):
    """Selection correct slot and inplementation in obj machine or in dict

    Parameters
    ----------
    self : ConvertMC
        A ConvertMC object
    """
    # check the direction of conversion
    if self.is_P_to_other == True:  # conversion to pyleecan
        motor_type = type(self.machine.stator.slot).__name__

        # selection type of Slot
        if motor_type == "SlotW11":
            name_slot = "Parallel_Tooth"

        elif motor_type == "":
            name_slot = ""

        else:
            raise Exception("Conversion of machine doesn't exist")

        # writting in dict
        if "[Calc_Options]" not in self.other_dict:
            self.other_dict["[Calc_Options]"] = {}
            temp_dict = self.other_dict["[Calc_Options]"]
            temp_dict["Slot_Type"] = name_slot
        else:
            self.other_dict["[Calc_Options]"]["Slot_Type"] = name_slot

    else:
        motor_type = self.other_dict["[Calc_Options]"]["Slot_Type"]

        # initialisation to set the slot in stator
        dict_machine = self.machine.stator.as_dict()
        self.machine.stator = LamSlotWind(init_dict=dict_machine)

        if motor_type == "Parallel_Tooth":
            # set the slot in obj machine, and add particularity to slotW11
            self.machine.stator.slot = SlotW11()
            self.machine.stator.slot.is_cstt_tooth = True
            self.machine.stator.slot.H1_is_rad = True
            self.machine.stator.is_internal = False

        else:
            raise Exception("Conversion of machine doesn't exist")
