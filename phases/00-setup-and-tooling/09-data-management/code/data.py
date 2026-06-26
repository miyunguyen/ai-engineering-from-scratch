import sys
import tempfile
import time
from pathlib import Path

try:
    from datasets import Dataset, load_dataset
except ImportError:
    print("Install the datasets library: pip install datasets")
    sys.exit(1)


def load_and_inspect(
    dataset_name: str = "glue", config: str = None, split: str = "train"
) -> Dataset:
    kwargs = {"path": dataset_name}
    if config:
        kwargs["name"] = config
    if split:
        kwargs["split"] = split

    ds = load_dataset(**kwargs)
    print(f"Dataset: {dataset_name}")
    print(f"  Rows: {len(ds)}")
    print(f"  Columns: {ds.column_names}")
    print(f"  Features: {ds.features}")
    print(f"  First 5 rows: {ds[:5]}")
    return ds


def stream_dataset(
    dataset_name: str,
    config: str = "en",
    split: str = "train",
    timeout: int = 10,
) -> int:
    kwargs = {"path": dataset_name, "split": split, "streaming": True}
    if config:
        kwargs["name"] = config

    ds = load_dataset(**kwargs)
    counter: int = 0

    start = time.monotonic()
    for _ in ds:
        counter += 1

        if time.monotonic() - start >= timeout:
            break

    print(f"Streamed {counter} rows from {dataset_name} in {timeout} seconds")
    return counter


def convert_format(ds, output_dir: str, name: str) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / f"{name}.csv"
    parquet_path = output_path / f"{name}.parquet"

    ds.to_csv(str(csv_path))
    ds.to_parquet(str(parquet_path))

    csv_size = csv_path.stat().st_size
    parquet_size = parquet_path.stat().st_size

    print(f"Format comparison for {name}:")
    print(f"  CSV:     {csv_size:>10,} bytes")
    print(f"  Parquet: {parquet_size:>10,} bytes")

    # Prevent potential division by zero
    if parquet_size > 0:
        print(f"  Parquet is {csv_size / parquet_size:.1f}x smaller than CSV")
    else:
        print("  Parquet size is 0 bytes; cannot calculate compression ratio.")

    return {"csv": csv_path, "parquet": parquet_path}


def make_splits(
    ds, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42
) -> dict:
    total = len(ds)
    train_size = int(round(total * train_ratio))
    val_size = int(round(total * val_ratio))
    test_size = total - train_size - val_size

    split1 = ds.train_test_split(test_size=test_size + val_size, seed=seed)
    train_ds = split1["train"]

    split2 = split1["test"].train_test_split(test_size=test_size, seed=seed)
    val_ds = split2["train"]
    test_ds = split2["test"]

    print(f"Splits (seed={seed}):")
    print(f"  Train: {len(train_ds):>6} ({len(train_ds)/total:.1%})")
    print(f"  Val:   {len(val_ds):>6} ({len(val_ds)/total:.1%})")
    print(f"  Test:  {len(test_ds):>6} ({len(test_ds)/total:.1%})")

    return {"train": train_ds, "val": val_ds, "test": test_ds}


if __name__ == "__main__":
    print("=" * 60)
    print("Data Management Exercise")
    print("=" * 60)

    print("\n--- 1. Load and inspect a dataset ---")
    ds = load_and_inspect("glue", "mrpc")

    print("\n--- 2. Stream a dataset ---")
    rows = stream_dataset("allenai/c4")

    print("\n--- 3. Convert formats ---")
    small_ds = ds.select(range(500))

    temp_dir = tempfile.gettempdir()
    path = convert_format(small_ds, temp_dir, "data_sample")

    print("\n--- 4. Make split ---")
    split = make_splits(small_ds, train_ratio=0.7, val_ratio=0.15, seed=42)

    print("\n" + "=" * 60)
    print("All checks passed. Your data pipeline is ready.")
    print("=" * 60)
