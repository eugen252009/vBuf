#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from qualify_weight_space_preparation import (
    Transform, codec_aware_permutation, hadamard, statistic_permutation,
    transform_metadata, vector_split,
)
from qualify_implicit_weight_feasibility import encode_fixed, reconstruct


class WeightSpacePreparationTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(7)
        self.w=self.rng.normal(size=(8,32)).astype(np.float32)
        self.x=self.rng.normal(size=(5,32)).astype(np.float32)

    def assert_equivalent(self,transform,tolerance=2e-5):
        reference=self.x@self.w.T
        prepared=transform.activations(self.x)@transform.weights(self.w).T
        np.testing.assert_allclose(prepared,reference,rtol=tolerance,atol=tolerance)

    def test_identity_baseline(self):
        transform=Transform("identity","identity")
        np.testing.assert_array_equal(transform.weights(self.w),self.w)
        np.testing.assert_array_equal(transform.activations(self.x),self.x)
        self.assert_equivalent(transform,0)

    def test_permutation_inverse_equivalence(self):
        permutation=self.rng.permutation(32).astype(np.int32)
        transform=Transform("p","permutation",permutation=permutation)
        inverse=np.argsort(permutation)
        np.testing.assert_array_equal(transform.activations(self.x)[:,inverse],self.x)
        self.assert_equivalent(transform)

    def test_signed_permutation_inverse(self):
        permutation=self.rng.permutation(32).astype(np.int32); signs=np.where(np.arange(32)%2,1.,-1.).astype(np.float32)
        transform=Transform("sp","signed_permutation",permutation=permutation,signs=signs)
        restored=(transform.activations(self.x)*signs[None,:])[:,np.argsort(permutation)]
        np.testing.assert_array_equal(restored,self.x)
        self.assert_equivalent(transform)

    def test_diagonal_scaling_inverse(self):
        scales=np.exp2((np.arange(32)%5)-2).astype(np.float32); transform=Transform("d","diagonal_scaling",scales=scales)
        np.testing.assert_array_equal(transform.activations(self.x)/scales[None,:],self.x)
        self.assert_equivalent(transform)

    def test_orthogonal_transform_inverse(self):
        for block in (8,16,32):
            transformed=hadamard(self.x,block); restored=hadamard(transformed,block)
            np.testing.assert_allclose(restored,self.x,rtol=2e-6,atol=2e-6)
            self.assert_equivalent(Transform(f"h{block}","fixed_orthogonal",hadamard_block=block))

    def test_metadata_accounting(self):
        transform=Transform("p","permutation",permutation=np.arange(32),metadata_bytes=8320)
        accounting=transform_metadata(transform,5_242_880)
        self.assertEqual(accounting["per_tensor_metadata_bytes"],8320)
        self.assertEqual(accounting["transformation_total_bytes"],8320)
        self.assertGreater(accounting["transformation_metadata_bpw"],0)

    def test_frozen_codec_identity_reproduction(self):
        flat=self.w.reshape(-1)
        left=encode_fixed(flat,"G2_CENTER_DENSE",16,batch_segments=4)
        right=encode_fixed(Transform("i","identity").weights(self.w).reshape(-1),"G2_CENTER_DENSE",16,batch_segments=4)
        np.testing.assert_array_equal(left.seeds,right.seeds); np.testing.assert_array_equal(reconstruct(left),reconstruct(right))

    def test_transformed_activation_handling(self):
        permutation=self.rng.permutation(32).astype(np.int32); transform=Transform("p","permutation",permutation=permutation)
        wrong=self.x@transform.weights(self.w).T; correct=transform.activations(self.x)@transform.weights(self.w).T
        self.assertGreater(np.linalg.norm(wrong-self.x@self.w.T),1e-3)
        np.testing.assert_allclose(correct,self.x@self.w.T,rtol=2e-5,atol=2e-5)

    def test_canonical_orientation(self):
        self.assertEqual(self.w.shape,(8,32)); self.assertEqual((self.x@self.w.T).shape,(5,8))

    def test_deterministic_optimization(self):
        weights=self.rng.normal(size=(128,64)).astype(np.float32); initial=statistic_permutation(weights)
        left,li=codec_aware_permutation(weights,initial,proposals=4); right,ri=codec_aware_permutation(weights,initial,proposals=4)
        np.testing.assert_array_equal(left,right); self.assertEqual(li,ri)

    def test_functional_test_isolation(self):
        qualification,test=vector_split(69); fit=qualification[::2]; validation=qualification[1::2]
        self.assertFalse(set(fit)&set(validation)); self.assertFalse(set(qualification)&set(test)); self.assertEqual((len(fit),len(validation),len(test)),(17,17,35))


if __name__=="__main__": unittest.main()
