import unittest
import bank


class TestHello(unittest.TestCase):
    def test_display(self):
        acc = bank.bank(0, "Patrick", 00000)
        self.assertEqual(acc.display(), 0)

    def test_withdraw(self):
        acc = bank.bank(0, "Patrick", 00000)
        acc.withdraw(10)
        self.assertEqual(acc.display(), -10)

    def test_deposit(self):
        acc = bank.bank(0, "Patrick", 00000)
        acc.deposit(10)
        self.assertEqual(acc.display(), 0)

    # def test_sin(self):
    #     self.assertEqual(hello.sin(0), 0)
    #     self.assertEqual(hello.sin(1), 0.8414709848078965)

    # def test_cos(self):
    #     self.assertEqual(hello.cos(0), 1)
    #     self.assertEqual(hello.cos(1), 0.5403023058681398)

    # def test_tan(self):
    #     self.assertEqual(hello.tan(0), 0)
    #     self.assertEqual(hello.tan(1), 1.5574077246549023)

    # def test_cot(self):
    #     self.assertEqual(hello.cot(0), float("inf"))
    #     self.assertEqual(hello.cot(1), 0.6420926159343306)


if __name__ == "__main__":
    unittest.main()
