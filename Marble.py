from enum import Enum


class Marble(Enum):
    none = -1
    Salt = 0
    Air = 1
    Fire = 2
    Water = 3
    Earth = 4
    Vitae = 5
    Mors = 6
    Quintessence = 7
    Quicksilver = 8
    Lead = 9
    Tin = 10
    Iron = 11
    Copper = 12
    Silver = 13
    Gold = 14

    def symbol(self):
        if self.value is self.none.value:
            return "-"
        if self.value in range(self.Quicksilver.value, self.Gold.value + 1):
            return self.name[0].upper()
        else:
            return self.name[0].lower()