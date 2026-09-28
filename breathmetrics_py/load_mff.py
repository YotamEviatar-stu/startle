import hashlib
import os

import numpy as np


def _pns_files(mff_path):
    from mne.io.egi.general import _get_signalfname
    all_files = _get_signalfname(mff_path)
    if "PNS" not in all_files:
        return None, [], []
    from defusedxml.minidom import parse
    pns_obj = parse(os.path.join(mff_path, "pnsSet.xml"))
    names, cals = [], []
    for sensor in pns_obj.getElementsByTagName("sensor"):
        names.append(sensor.getElementsByTagName("name")[0].firstChild.data)
        unit_node = sensor.getElementsByTagName("unit")[0].firstChild
        unit = unit_node.data if unit_node is not None else ""
        cals.append(1e-6 if unit == "uV" else 1.0)
    return os.path.join(mff_path, all_files["PNS"]["signal"]), names, cals


def list_channels(mff_path):
    bin_path, names, _ = _pns_files(mff_path)
    if bin_path is not None:
        return names
    import mne
    raw = mne.io.read_raw_egi(mff_path, preload=False, verbose="error")
    return list(raw.ch_names)


def _read_pns_channel(bin_path, ch_index):
    from mne.io.egi.general import _block_r

    size = os.path.getsize(bin_path)
    chunks = []
    block_size = size
    n_channels = None
    sfreq = None
    with open(bin_path, "rb", buffering=1 << 20) as fid:
        pos = 0
        while pos < size:
            block = _block_r(fid)
            if block is not None:
                block_size = block["block_size"]
                n_channels = block["nc"]
                sfreq = block["sfreq"]
            if n_channels is None:
                raise RuntimeError("first block of %s has no header" % bin_path)
            n_samples = block_size // 4 // n_channels
            data = np.fromfile(fid, "<f4", block_size // 4)
            chunks.append(data.reshape(n_channels, n_samples)[ch_index].copy())
            pos = fid.tell()

    return np.concatenate(chunks).astype(float), float(sfreq)


def epoch_clock(mff_path):
    from defusedxml.minidom import parse
    epochs = parse(os.path.join(mff_path, "epochs.xml")).getElementsByTagName("epoch")
    begin = np.array([int(e.getElementsByTagName("beginTime")[0].firstChild.data)
                      for e in epochs]) / 1e6
    end = np.array([int(e.getElementsByTagName("endTime")[0].firstChild.data)
                    for e in epochs]) / 1e6
    start = np.concatenate([[0.0], np.cumsum(end - begin)[:-1]])
    offset = begin - begin[0] - start

    def to_clock(t):
        t = np.asarray(t, float)
        k = np.clip(np.searchsorted(start, t, side="right") - 1, 0, None)
        return t + offset[k]
    return to_clock


def _resample(signal, srate, target_srate):
    from scipy.signal import decimate, resample_poly
    from fractions import Fraction

    if target_srate is None or abs(srate - target_srate) <= 1e-6:
        return signal, srate
    if target_srate > srate:
        raise ValueError("refusing to upsample %g -> %g Hz" % (srate, target_srate))
    ratio = srate / target_srate
    if abs(ratio - round(ratio)) < 1e-9:
        q = int(round(ratio))
        out = signal
        while q > 1:
            step = min(q, 10)
            while q % step:
                step -= 1
            out = decimate(out, step, ftype="iir", zero_phase=True)
            q //= step
        return np.asarray(out, dtype=float), float(target_srate)
    frac = Fraction(target_srate / srate).limit_denominator(1000)
    out = resample_poly(signal, frac.numerator, frac.denominator)
    return np.asarray(out, dtype=float), float(target_srate)


def _cache_path(cache_dir, mff_path, channel, target_srate, crop_sec):
    key = "%s|%s|%s|%s" % (os.path.abspath(mff_path), channel, target_srate, crop_sec)
    stamp = hashlib.sha1(key.encode()).hexdigest()[:12]
    base = os.path.basename(os.path.normpath(mff_path)).replace(".mff", "")
    return os.path.join(cache_dir, "%s_%s_%s.npz" % (base, str(channel).replace("/", "_"), stamp))


def load_mff(mff_path, channel, target_srate=None, crop_sec=None, cache_dir=None):
    """Load ONE channel.

    For a PNS/PIB channel (Airflow, belts, ...) this reads only
    signal2.bin directly and never touches the multi-GB EEG signal1.bin,
    which is what mne.io.read_raw_egi does even when a single channel is
    picked. Falls back to MNE for EEG/stim channels.
    """
    cache_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = _cache_path(cache_dir, mff_path, channel, target_srate, crop_sec)
        if os.path.exists(cache_file):
            z = np.load(cache_file)
            return z["signal"], float(z["srate"])

    bin_path, pns_names, pns_cals = _pns_files(mff_path)
    use_pns = bin_path is not None and (
        (isinstance(channel, int) and 0 <= channel < len(pns_names))
        or channel in pns_names)

    if use_pns:
        idx = channel if isinstance(channel, int) else pns_names.index(channel)
        signal, srate = _read_pns_channel(bin_path, idx)
        signal *= pns_cals[idx]
        if crop_sec is not None:
            lo = int(round(float(crop_sec[0]) * srate))
            hi = int(round(float(crop_sec[1]) * srate)) + 1
            signal = signal[lo:hi]
    else:
        signal, srate = _load_mff_mne(mff_path, channel, crop_sec)

    signal, srate = _resample(signal, srate, target_srate)

    if cache_file:
        np.savez_compressed(cache_file, signal=signal, srate=srate)
    return signal, srate


def _load_mff_mne(mff_path, channel, crop_sec):
    import mne
    raw = mne.io.read_raw_egi(mff_path, preload=False, verbose="error")
    name = raw.ch_names[channel] if isinstance(channel, int) else channel
    if name not in raw.ch_names:
        raise ValueError("channel %r not in file; see list_channels()" % channel)
    if crop_sec is not None:
        raw.crop(tmin=float(crop_sec[0]), tmax=float(crop_sec[1]))
    raw.pick([name])
    raw.load_data(verbose="error")
    return raw.get_data(picks=[name])[0].astype(float), float(raw.info["sfreq"])


def load_array(path, srate):
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path).astype(float).ravel(), float(srate)
    return np.loadtxt(path, delimiter=",").astype(float).ravel(), float(srate)
