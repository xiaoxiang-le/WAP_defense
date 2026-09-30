from sklearn.model_selection import train_test_split

from loaddata import load_all_data


def _deduplicate(data, labels):
    samples = {}
    conflicts = set()
    for payload, label in zip(data, labels):
        key = payload.strip()
        previous = samples.get(key)
        if previous is not None and previous != label:
            conflicts.add(key)
        else:
            samples[key] = label

    for key in conflicts:
        samples.pop(key, None)
    return list(samples.keys()), list(samples.values())


def splitmain(seed=42):
    data, labels = load_all_data()
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
    return (
        train_data,
        train_labels,
        validation_data,
        validation_labels,
        test_data,
        test_labels,
    )
