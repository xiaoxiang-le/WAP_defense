import hashlib

from sklearn.model_selection import train_test_split

from loaddata import load_all_data
from staticfeature import normalize_payload


def _deduplicate(data, labels):
    samples = {}
    conflicts = set()
    for payload, label in zip(data, labels):
        original = payload.strip()
        key = normalize_payload(original).strip()
        if not key:
            continue
        previous = samples.get(key)
        if previous is not None and previous[1] != label:
            conflicts.add(key)
        else:
            samples[key] = (original, label)

    for key in conflicts:
        samples.pop(key, None)
    ordered = [samples[key] for key in sorted(samples)]
    return [item[0] for item in ordered], [item[1] for item in ordered]


def _dataset_fingerprint(data, labels):
    digest = hashlib.sha256()
    for payload, label in zip(data, labels):
        digest.update(normalize_payload(payload).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(label).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def splitmain(seed=42, return_metadata=False):
    data, labels = load_all_data()
    raw_sample_count = len(data)
    data, labels = _deduplicate(data, labels)

    train_data, test_data, train_labels, test_labels = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    train_data, validation_data, train_labels, validation_labels = train_test_split(
        train_data,
        train_labels,
        test_size=0.25,
        random_state=seed,
        stratify=train_labels,
    )

    print(
        "数据集大小：训练 {}，验证 {}，测试 {}".format(
            len(train_data), len(validation_data), len(test_data)
        )
    )
    result = (
        train_data,
        train_labels,
        validation_data,
        validation_labels,
        test_data,
        test_labels,
    )
    if return_metadata:
        metadata = {
            "dataset_fingerprint": _dataset_fingerprint(data, labels),
            "raw_sample_count": raw_sample_count,
            "sample_count": len(data),
            "train_count": len(train_data),
            "validation_count": len(validation_data),
            "test_count": len(test_data),
        }
        return result + (metadata,)
    return result
