"""RGI entities grouped into complexes through the RGI 6.2 intersects."""

from data_processing.glacier_utils import rgi_complexes


def test_mera_is_one_complex():
    # Mera is cut into seven RGI 6 entities sharing divides; the two branches of Urumqi
    # Glacier No. 1 touch no other entity
    mera = [
        "RGI60-15.03581",
        "RGI60-15.03583",
        "RGI60-15.03584",
        "RGI60-15.03585",
        "RGI60-15.03586",
        "RGI60-15.03591",
        "RGI60-15.03592",
    ]
    urumqi = ["RGI60-13.45334", "RGI60-13.45335"]
    complexes = rgi_complexes(mera + urumqi)
    assert {complexes[g] for g in mera} == {"RGI60-15.03581"}
    assert [complexes[g] for g in urumqi] == urumqi
