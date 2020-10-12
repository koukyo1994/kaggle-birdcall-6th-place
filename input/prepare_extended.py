import argparse
import warnings

import librosa
import pandas as pd
import soundfile as sf

from pathlib import Path
from joblib import delayed, Parallel


def resample(df: pd.DataFrame, target_sr: int, audio_dir: Path):
    resample_dir = Path("train_audio_resampled")
    warnings.simplefilter("ignore")

    for i, row in df.iterrows():
        ebird_code = row.ebird_code
        filename = row.filename
        ebird_dir = resample_dir / ebird_code
        if not ebird_dir.exists():
            ebird_dir.mkdir(exist_ok=True, parents=True)

        try:
            y, _ = librosa.load(audio_dir / ebird_code / filename, sr=target_sr,
                                res_type="kaiser_fast",
                                mono=True)
            filename = filename.replace(".mp3", ".wav")
            sf.write(ebird_dir / filename, y, samplerate=target_sr)
        except Exception:
            with open("skipped.txt", "a") as f:
                file_path = str(audio_dir / ebird_code / filename)
                f.write(file_path + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", default=32000, type=int)
    parser.add_argument("--n_splits", default=12, type=int)
    args = parser.parse_args()

    target_sr = args.sr

    a_m = pd.read_csv("train_extended_a_m.csv")
    n_z = pd.read_csv("train_extended_n_z.csv")
    a_m_dir = Path("A-M")
    n_z_dir = Path("N-Z")
    for train, audio_dir in zip([a_m, n_z], [a_m_dir, n_z_dir]):
        dfs = []
        for i in range(args.n_splits):
            if i == args.n_splits - 1:
                start = i * (len(train) // args.n_splits)
                df = train.iloc[start:, :].reset_index(drop=True)
                dfs.append(df)
            else:
                start = i * (len(train) // args.n_splits)
                end = (i + 1) * (len(train) // args.n_splits)
                df = train.iloc[start:end, :].reset_index(drop=True)
                dfs.append(df)

        Parallel(
            n_jobs=args.n_splits,
            verbose=10)(delayed(resample)(df, args.sr, audio_dir) for df in dfs)

    a_m["resampled_sampling_rate"] = target_sr
    a_m["resampled_filename"] = a_m["filename"].map(
        lambda x: x.replace(".mp3", ".wav"))
    a_m["resampled_channels"] = "1 (mono)"

    n_z["resampled_sampling_rate"] = target_sr
    n_z["resampled_filename"] = n_z["filename"].map(
        lambda x: x.replace(".mp3", ".wav"))
    n_z["resampled_channels"] = "1 (mono)"

    concat = pd.concat([a_m, n_z], axis=0, sort=False).reset_index(drop=True)
    concat.to_csv("train_extended.csv", index=False)
