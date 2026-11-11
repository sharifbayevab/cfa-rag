"""Scientific protocol checks, not tests of historical model quality."""
import itertools
import unittest

import numpy as np

from rebuild import random_jaccard
from selector_oof import load_aligned, validate_split


class ProtocolTests(unittest.TestCase):
    def test_random_reference_matches_enumeration(self):
        # Exact enumeration protects against comparing Jaccard with E[overlap]
        # divided by E[union], which is a different null quantity.
        for m in range(2, 8):
            for s in range(m+1):
                causal = set(range(s))
                for k in range(1, m+1):
                    values=[]
                    for subset in itertools.combinations(range(m), k):
                        selected=set(subset)
                        values.append(len(causal&selected)/len(causal|selected))
                    self.assertAlmostEqual(random_jaccard(m,s,k),float(np.mean(values)),places=12)

    def test_rejects_same_question_across_different_rows(self):
        groups=np.array(["hotpot:a", "hotpot:b", "hotpot:a", "2wiki:a"])
        with self.assertRaisesRegex(ValueError,"Question leakage"):
            validate_split(groups,np.array([0,1]),np.array([2,3]))
        # Dataset-prefixed IDs distinguish questions whose local IDs coincide.
        validate_split(groups,np.array([0,2]),np.array([1,3]))

    def test_real_feature_alignment(self):
        X,y,ds,gen,qid,idx,names=load_aligned()
        self.assertEqual(len(set(qid)),450)
        self.assertEqual(len(set(gen)),5)
        self.assertEqual(X.shape,(29975,9))
        self.assertEqual(len(y),len(idx))
        self.assertEqual(len(names),9)


if __name__=="__main__":unittest.main()
