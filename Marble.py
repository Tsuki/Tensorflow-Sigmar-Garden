from enum import Enum


class Marble(Enum):
    none = 0
    Salt = 1
    Air = 2
    Fire = 3
    Water = 4
    Earth = 5
    Vitae = 6
    Mors = 7
    Quintessence = 8
    Quicksilver = 9
    Lead = 10
    Tin = 11
    Iron = 12
    Copper = 13
    Silver = 14
    Gold = 15

    def symbol(self):
        if self.value is self.none.value:
            return "-"
        if self.value in range(self.Quicksilver.value, self.Gold.value + 1):
            return self.name[0].upper()
        else:
            return self.name[0].lower()
