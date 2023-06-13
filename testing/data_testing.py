import unittest


class FiberPhotometryDataTest(unittest.TestCase):
    def test_import(self):
        try:
            from src.FiberPhotometry_AE_bugsiesegal.data import FiberPhotometryDataset
        except ImportError:
            self.fail("Was not able to import FiberPhotometryDataset")

    def test_instantiation(self):
        try:
            from src.FiberPhotometry_AE_bugsiesegal.data import FiberPhotometryDataset
            import torch
            import numpy as np
            dataset = FiberPhotometryDataset(torch.randn(1000), torch.randn(1000), 100, 10)
            dataset = FiberPhotometryDataset(np.random.randn(1000), np.random.randn(1000), 100, 10)
        except ImportError:
            self.fail("Was not able to import FiberPhotometryDataset")
        except TypeError:
            self.fail("Was not able to instantiate FiberPhotometryDataset")

    def test_normalize_windows(self):
        try:
            from src.FiberPhotometry_AE_bugsiesegal.data import FiberPhotometryDataset
            import torch
            import numpy as np
            dataset = FiberPhotometryDataset(torch.randn(1000), torch.randn(1000), 100, 10)
            dataset.normalize_windows()
            dataset = FiberPhotometryDataset(np.random.randn(1000), np.random.randn(1000), 100, 10)
            dataset.normalize_windows()
        except ImportError:
            self.fail("Was not able to import FiberPhotometryDataset")
        except TypeError:
            self.fail("Was not able to instantiate FiberPhotometryDataset")
        except AttributeError:
            self.fail("Was not able to normalize windows")

    def test_len(self):
        try:
            from src.FiberPhotometry_AE_bugsiesegal.data import FiberPhotometryDataset
            import torch
            import numpy as np
            dataset = FiberPhotometryDataset(torch.randn(1000), torch.randn(1000), 100, 10)
            dataset.__len__()
            dataset = FiberPhotometryDataset(np.random.randn(1000), np.random.randn(1000), 100, 10)
            dataset.__len__()
        except ImportError:
            self.fail("Was not able to import FiberPhotometryDataset")
        except TypeError:
            self.fail("Was not able to instantiate FiberPhotometryDataset")
        except AttributeError:
            self.fail("Was not able to get length of dataset")

    def test_get_sample_times(self):
        try:
            from src.FiberPhotometry_AE_bugsiesegal.data import FiberPhotometryDataset
            import torch
            import numpy as np
            dataset = FiberPhotometryDataset(torch.randn(1000), torch.randn(1000), 100, 10)
            dataset.get_sample_times(0)
            dataset = FiberPhotometryDataset(np.random.randn(1000), np.random.randn(1000), 100, 10)
            dataset.get_sample_times(0)
        except ImportError:
            self.fail("Was not able to import FiberPhotometryDataset")
        except TypeError:
            self.fail("Was not able to instantiate FiberPhotometryDataset")
        except AttributeError:
            self.fail("Was not able to get sample times")

    def test_getitem(self):
        try:
            from src.FiberPhotometry_AE_bugsiesegal.data import FiberPhotometryDataset
            import torch
            import numpy as np
            dataset = FiberPhotometryDataset(torch.randn(1000), torch.randn(1000), 100, 10)
            dataset.__getitem__(0)
            dataset = FiberPhotometryDataset(np.random.randn(1000), np.random.randn(1000), 100, 10)
            dataset.__getitem__(0)
        except ImportError:
            self.fail("Was not able to import FiberPhotometryDataset")
        except TypeError:
            self.fail("Was not able to instantiate FiberPhotometryDataset")
        except AttributeError:
            self.fail("Was not able to get item from dataset")


if __name__ == '__main__':
    unittest.main()
