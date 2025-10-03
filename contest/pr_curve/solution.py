import numpy as np


def build_precision_recall_curve(
    true_labels: np.ndarray, predicted_probas: np.ndarray
) -> np.ndarray:
    """
    Данная функция строит PR-кривую для задачи бинарной классификации.
    В случае, когда нет ни одного объекта положительного класса функция должна вызывать ValueError().

    Args:
        true_labels (np.ndarray): Массив истинных меток класса. Состоит из 0 и 1.
            1 считается меткой положительного класса.
        predicted_probas (np.ndarray): Массив предсказанных вероятностей принадлежности объекта
            к положительному классу.

    Returns:
        np.ndarray: Массив размерами (len(true_labels)+1, 2), где в каждой строчке стоит пара (precision, recall), первым элементом всегда идет (0, 1)
    """
    if not np.any(true_labels):
        raise ValueError()

    pr_curve = [[0.0, 1.0]]
    for threshold in np.sort(predicted_probas)[::-1]:
        predicted_labels = predicted_probas >= threshold

        tp = np.sum(predicted_labels & true_labels)
        fp = np.sum(predicted_labels & (~true_labels))
        fn = np.sum((~predicted_labels) & true_labels)

        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn)

        pr_curve.append([r, p])

    return np.asarray(pr_curve, dtype=float)