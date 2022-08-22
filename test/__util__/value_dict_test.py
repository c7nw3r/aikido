from unittest import TestCase

from aikido.__util__.value_dict import ValueDict


class ValueDictTest(TestCase):

    def test_abc(self):
        value_dict = ValueDict({"a": 0, "b": 1, "c": 2})
        self.assertEqual([0, 1, 2], list(value_dict))
