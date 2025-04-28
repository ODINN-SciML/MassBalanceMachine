import os
import massbalancemachine as mbm
from regions.Switzerland.scripts.glamos_preprocess import get_geodetic_MB

if "CI" in os.environ:
    pathDataDownload = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../dataDownload/data/'))
    dataPath = pathDataDownload
else:
    dataPath = None

def test_geodetic_data():
    cfg = mbm.SwitzerlandConfig(dataPath=dataPath)

    geodetic_mb = get_geodetic_MB(cfg)
    print("geodetic_mb.shape=",geodetic_mb.shape)
    assert geodetic_mb.shape == (331, 17)

if __name__ == "__main__":
    test_geodetic_data()
