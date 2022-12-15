import unittest
import rqcopt as oc


class TestSplittingMethod(unittest.TestCase):

    def test_suzuki(self):
        """
        Consistency checks for the Suzuki splitting methods.
        """
        for nterms in range(2, 4):
            for k in range(1, 3):
                rule = oc.SplittingMethod.suzuki(nterms, k)
                # more refined consistency checks are performed within __init__
                self.assertAlmostEqual(sum(rule.coeffs), rule.nterms)

    def test_yoshida(self):
        """
        Consistency checks for the splitting method by Yoshida of order 4.
        """
        for nterms in [2, 3]:
            rule = oc.SplittingMethod.yoshida4(nterms)
            # more refined consistency checks are performed within __init__
            self.assertAlmostEqual(sum(rule.coeffs), rule.nterms)

    def test_mclachlan(self):
        """
        Consistency checks for splitting methods by Robert I. McLachlan.
        """
        for rule in [oc.SplittingMethod.mclachlan4(m) for m in [4, 5]]:
            # more refined consistency checks are performed within __init__
            self.assertAlmostEqual(sum(rule.coeffs), rule.nterms)

    def test_blanes_moan(self):
        """
        Consistency checks for splitting method by Blanes and Moan.
        """
        rule = oc.SplittingMethod.blanes_moan()
        # more refined consistency checks are performed within __init__
        self.assertAlmostEqual(sum(rule.coeffs), rule.nterms)

    def test_auzinger(self):
        """
        Consistency checks for AY 15-6 splitting method by Auzinger, Hofstätter, Ketcheson, Koch.
        """
        rule = oc.SplittingMethod.auzinger15_6()
        # more refined consistency checks are performed within __init__
        self.assertAlmostEqual(sum(rule.coeffs), rule.nterms)


if __name__ == "__main__":
    unittest.main()
