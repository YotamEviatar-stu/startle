import sys
from pathlib import Path

import mne
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from caracas.run import list_sessions

CACHE = Path(__file__).resolve().parents[1] / "_cache" / "iclabel"
CLASSES = ("brain", "muscle artifact", "eye blink", "heart beat", "line noise", "channel noise", "other")


def run_session(subject, session, mff):
    raw = mne.io.read_raw_egi(mff, preload=False, verbose="error")
    raw.pick("eeg").load_data()
    raw.filter(l_freq=1.0, h_freq=100.0)
    raw.set_eeg_reference("average")
    ica = ICA(n_components=50, max_iter="auto", method="infomax", random_state=97,
              fit_params=dict(extended=True))
    ica.fit(raw)
    proba = iclabel_label_components(raw, ica, backend="onnx")
    heart = int(np.argmax(proba[:, CLASSES.index("heart beat")]))
    signal = ica.get_sources(raw).get_data(picks=[heart])[0].astype(np.float32)
    out_dir = CACHE / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    ica.save(out_dir / f"{mff.stem}-ica.fif", overwrite=True)
    np.savez(out_dir / f"{mff.stem}.npz", subject=subject, session=session, mff=mff.name,
             signal=signal, sfreq=raw.info["sfreq"], heart_idx=heart, proba=proba,
             classes=np.array(CLASSES))
    return raw, ica, proba, heart


def main(only):
    for subject, session, mff in list_sessions():
        if f"{subject}/{session}" in only:
            run_session(subject, session, mff)


if __name__ == "__main__":
    main(sys.argv[1:])
