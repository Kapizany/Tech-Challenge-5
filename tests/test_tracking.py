import hashlib

from adaptive_offers.tracking import lineage, mlflow_tags


def test_lineage_hashes_data_and_formats_tags(tmp_path):
    data = tmp_path / "data.csv"
    data.write_bytes(b"a,b\n1,2\n")
    values = lineage(data)
    assert values["data_sha256"] == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    tags = mlflow_tags({**values, "git_dirty": True, "missing": None})
    assert tags["git_dirty"] == "true"
    assert "missing" not in tags
