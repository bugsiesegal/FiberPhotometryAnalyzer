import unittest


class AutoEncoderTest(unittest.TestCase):
    def test_import(self):
        try:
            from src.FiberPhotometry_AE_bugsiesegal.autoencoder import AutoEncoder
        except ImportError:
            self.fail("Was not able to import AutoEncoder")

    def test_instantiation(self):
        try:
            from src.FiberPhotometry_AE_bugsiesegal.autoencoder import AutoEncoder
            import torch
            import numpy as np
            autoencoder = AutoEncoder(100, 10)
        except ImportError:
            self.fail("Was not able to import AutoEncoder")
        except TypeError:
            self.fail("Was not able to instantiate AutoEncoder")

    def test_train(self):
        try:
            from src.FiberPhotometry_AE_bugsiesegal.autoencoder import AutoEncoder
            import torch
            import numpy as np
            autoencoder = AutoEncoder(100, 10)
            autoencoder.fit(torch.randn(100))
            autoencoder = AutoEncoder(100, 10)
            autoencoder.fit(np.random.randn(100))
        except ImportError:
            self.fail("Was not able to import AutoEncoder")
        except TypeError:
            self.fail("Was not able to instantiate AutoEncoder")
        except AttributeError:
            self.fail("Was not able to train AutoEncoder")


if __name__ == '__main__':
    unittest.main()
