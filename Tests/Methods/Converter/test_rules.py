import pytest

from pyleecan.Classes.RuleSimple import RuleSimple
from pyleecan.Classes.RuleEquation import RuleEquation
from pyleecan.Classes.RuleComplex import RuleComplex

from pyleecan.Methods.Converter.RuleSimple.convert_to_P import convert_to_P
from pyleecan.Classes.MachineSIPMSM import MachineSIPMSM


from pyleecan.Classes.ConvertMC import ConvertMC

rule_list = list()


rule_list.append(RuleComplex(fct_name="parallel_slot_slotW21", folder="pyleecan"))

rule_list.append(
    RuleSimple(
        other_key_list=["[Dimensions]", "Slot_Opening"],
        P_obj_path=f"machine.stator.slot.W0",
        unit_type="m",
        scaling_to_P=1,
    )
)

rule_list.append(
    RuleSimple(
        other_key_list=["[Dimensions]", "Slot_tooth"],
        P_obj_path=f"machine.stator.slot.W2",
        unit_type="m",
        scaling_to_P=0.5,
    )
)

rule_list.append(
    RuleEquation(
        param=[
            {
                "src": "other",
                "path": ["[Dimensions]", "Slot_Depth"],
                "variable": "y",
            },
            {
                "src": "pyleecan",
                "path": f"machine.stator.slot.H2",
                "variable": "x",
            },
            {
                "src": "pyleecan",
                "path": f"machine.stator.slot.W0",
                "variable": "b",
            },
        ],
        unit_type="m",
        scaling_to_P="y/3 = b +2*x",
    )
)

rule_list.append(
    RuleEquation(
        param=[
            {
                "src": "other",
                "path": ["[Dimensions]", "Slot_Depth"],
                "variable": "d",
            },
            {
                "src": "other",
                "path": ["[Dimensions]", "Slot_2"],
                "variable": "y",
            },
            {
                "src": "other",
                "path": ["[Dimensions]", "Slot_3"],
                "variable": "e",
            },
            {
                "src": "other",
                "path": ["[Dimensions]", "Slot_4"],
                "variable": "f",
            },
            {
                "src": "pyleecan",
                "path": f"machine.stator.slot.H2",
                "variable": "a",
            },
            {
                "src": "pyleecan",
                "path": f"machine.stator.slot.W0",
                "variable": "b",
            },
            {
                "src": "pyleecan",
                "path": f"machine.stator.slot.W1",
                "variable": "x",
            },
        ],
        unit_type="m",
        scaling_to_P="(y/3 - e )/f +d  = b +2*x -a ",
    )
)


# add equation rules

other_dict = {}
other_dict["[Dimensions]"] = {}
other_dict["[Dimensions]"]["Slot_tooth"] = 15
other_dict["[Dimensions]"]["Slot_Opening"] = 12.5
other_dict["[Dimensions]"]["Slot_Depth"] = 72
other_dict["[Dimensions]"]["Slot_2"] = 6.75
other_dict["[Dimensions]"]["Slot_3"] = 6.75
other_dict["[Dimensions]"]["Slot_4"] = 3.25


class Test_converter_mot(object):
    def compare_rule(self):
        machine = MachineSIPMSM()

        # first rule complex use to define a slot
        rule = rule_list[0]
        machine = rule.convert_to_P(other_dict, machine)
        assert type(machine.stator.slot).__name__ == "SlotW21"

        # rule simple to set slot.W0
        rule = rule_list[1]
        machine = rule.convert_to_P(other_dict, machine)
        msg = f"{machine.stator.slot.W0}, should be equal at 12.5"
        assert abs(machine.stator.slot.W0) == pytest.approx(12.5), msg

        # rule simple to set value, with conversion
        rule = rule_list[2]
        machine = rule.convert_to_P(other_dict, machine)
        msg = f"{machine.stator.slot.W2}, should be equal at 7.5"
        assert abs(machine.stator.slot.W2) == pytest.approx(7.5), msg

        # rule equation
        rule = rule_list[3]
        machine = rule.convert_to_P(other_dict, machine)
        msg = f"{machine.stator.slot.H2}, should be equal at 5.75"
        assert abs(machine.stator.slot.H2) == pytest.approx(5.75), msg

        # rule equation
        rule = rule_list[4]
        machine = rule.convert_to_P(other_dict, machine)
        msg = machine.stator.slot.W1
        assert abs(machine.stator.slot.W1) == pytest.approx(31.9326923076923), msg


if __name__ == "__main__":
    a = Test_converter_mot()
    a.compare_rule()
    print("Done")
