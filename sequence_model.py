"""Baseline transparan untuk analisis urutan angka 4 digit; bukan jaminan prediksi."""
from collections import Counter
import re

_DIGITS = re.compile(r"[0-9]{4}")


def validate_numbers(values):
    """Return 4-character strings and preserve leading zeroes."""
    result = []
    for value in values:
        item = str(value).strip()
        if not _DIGITS.fullmatch(item):
            raise ValueError(f"Angka harus terdiri dari tepat 4 digit: {item!r}")
        result.append(item)
    return result


def _validate_minimum_history(minimum_history):
    if isinstance(minimum_history, bool) or not isinstance(minimum_history, int) or minimum_history < 1:
        raise ValueError("minimum_history harus berupa bilangan bulat positif.")


def predict_next(values, minimum_history=5):
    """Choose the most frequent digit at each position.

    Ties are resolved by the most recently observed digit among tied digits.
    This baseline does not establish that historical frequencies predict
    independent or random outcomes.
    """
    _validate_minimum_history(minimum_history)
    data = validate_numbers(values)
    if len(data) < minimum_history:
        return None

    candidate = []
    for position in range(4):
        counts = Counter(item[position] for item in data)
        highest = max(counts.values())
        tied = {digit for digit, count in counts.items() if count == highest}
        candidate.append(next(item[position] for item in reversed(data) if item[position] in tied))
    return "".join(candidate)


def walk_forward_backtest(values, minimum_history=5):
    """Evaluate historical guesses using only observations before each guess."""
    _validate_minimum_history(minimum_history)
    data = validate_numbers(values)
    records = []
    correct = 0
    for index in range(minimum_history, len(data)):
        guess = predict_next(data[:index], minimum_history)
        actual = data[index]
        hit = guess == actual
        correct += int(hit)
        records.append({"history_size": index, "predicted": guess, "actual": actual, "exact_match": hit})
    tested = len(records)
    return {"tested": tested, "correct": correct, "accuracy": correct / tested if tested else None, "records": records}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("numbers", nargs="*", help="Angka 4 digit, urut lama ke baru")
    args = parser.parse_args()
    if not args.numbers:
        parser.error("Masukkan angka 4 digit dalam urutan lama ke baru.")
    data = validate_numbers(args.numbers)
    guess = predict_next(data)
    report = walk_forward_backtest(data)
    print(f"Jumlah data: {len(data)}")
    print(f"Kandidat berikutnya: {guess or 'belum cukup data'}")
    if report["accuracy"] is None:
        print("Backtest: belum cukup data")
    else:
        print(f"Backtest tepat 4 digit: {report['correct']}/{report['tested']} ({report['accuracy']:.2%})")
