"""Run validation test for cellsignal module."""
import os
import argparse
import time
import unittest
import torch
import pytorch_lightning as pl

from cellsignal.data import RxRx1DataModule

class TestEvaluateCellSignalClassifier(unittest.TestCase):
    """Evaluate CellSignal Classifier on the test set """
    @torch.no_grad()
    def test_evaluate_cellsignal_classifier(self):
        dataset = RxRx1DataModule(args=argparse.Namespace(gpus=None, batch_size=128, num_workers=0))
        dataset.prepare_data()
        dataset.setup()

        ## TODO: Add model