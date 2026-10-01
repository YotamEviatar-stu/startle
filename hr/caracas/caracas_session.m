function caracas_session(mff_path, out_mat, deps_dir)

home = getenv('HOME');
ftdir = fullfile(home, 'Documents', 'MATLAB', 'fieldtrip');
sasdir = fullfile(home, 'code', 'tools', 'SASICA');

restoredefaultpath;
addpath(ftdir);
ft_defaults;
addpath(fullfile(ftdir, 'external', 'eeglab'));
addpath(sasdir);
addpath(fullfile(sasdir, 'CARACAS'));
addpath(fullfile(sasdir, 'CARACAS', 'heart_functions'));
addpath(fullfile(sasdir, 'eeglab'));
addpath(genpath(fullfile(sasdir, 'eeglab', 'functions')));
addpath(fullfile(sasdir, 'eeglab', 'plugins', 'firfilt'));
if ~exist(fullfile(deps_dir, 'zscore.m'), 'file')
    if ~exist(deps_dir, 'dir'), mkdir(deps_dir); end
    copyfile(fullfile(sasdir, 'private', 'zscore.m'), fullfile(deps_dir, 'zscore.m'));
end
addpath(deps_dir);
ft_warning('off', 'FieldTrip:dataContainsNaN');

hdr = ft_read_header(mff_path, 'headerformat', 'egi_mff_v1');
eegchan = hdr.label(strcmp(hdr.chantype, 'eeg'));

cfg = [];
cfg.dataset = mff_path;
cfg.headerformat = 'egi_mff_v1';
cfg.dataformat = 'egi_mff_v1';
cfg.channel = eegchan;
data = ft_preprocessing(cfg);
data = preprocess_blockwise(data, 32);

rank_data = numel(data.label) - 1;
ncomp = round(rank_data / 5);

cfg = [];
cfg.channel = 'all';
cfg.numcomponent = ncomp;
cfg.runica.extended = 1;
rng(123);
comp = ft_componentanalysis(cfg, data);
clear data

elec = ft_read_sens(fullfile(ftdir, 'template', 'electrode', 'GSN-HydroCel-257.sfp'), 'senstype', 'eeg');
EEG = eeg_emptyset;
EEG.setname = 'internal';
EEG.nbchan = numel(comp.topolabel);
EEG.trials = 1;
EEG.pnts = size(comp.trial{1}, 2);
EEG.srate = comp.fsample;
EEG.xmin = comp.time{1}(1);
EEG.xmax = comp.time{1}(end);
EEG.times = comp.time{1} * 1000;
EEG.icaact = comp.trial{1};
EEG.icawinv = comp.topo;
EEG.icaweights = comp.unmixing;
EEG.icasphere = eye(size(EEG.icaweights, 2));
EEG.chanlocs = struct();
for i = 1:EEG.nbchan
    lab = comp.topolabel{i};
    EEG.chanlocs(i).labels = lab;
    if strcmp(lab, 'VREF'), lab = 'Cz'; end
    ichan = find(strcmp(elec.label, lab));
    if ~isempty(ichan)
        EEG.chanlocs(i).X = elec.chanpos(ichan, 1);
        EEG.chanlocs(i).Y = elec.chanpos(ichan, 2);
        EEG.chanlocs(i).Z = elec.chanpos(ichan, 3);
    end
end
EEG.chanlocs = convertlocs(EEG.chanlocs, 'cart2all');
EEG.chaninfo.nosedir = '+X';
EEG.data = EEG.icawinv * EEG.icaact;
EEG.icachansind = 1:EEG.nbchan;

cfg_SASICA = SASICA('getdefs');
mnames = fieldnames(cfg_SASICA);
for m = 1:numel(mnames)
    if isstruct(cfg_SASICA.(mnames{m})) && isfield(cfg_SASICA.(mnames{m}), 'enable')
        cfg_SASICA.(mnames{m}).enable = false;
    end
end
cfg_SASICA.CARACAS.enable = true;
cfg_SASICA.opts.noplot = 1;
cfg_SASICA.opts.noplotselectcomps = 1;
EEG = eeg_SASICA(EEG, cfg_SASICA);

meas = EEG.reject.SASICA.icaCARACAS;
caracas_cfg = meas(1).cfg;
meas = rmfield(meas, 'cfg');
is_cardiac = EEG.reject.SASICA.icarejCARACAS;
cardiac_idx = find(is_cardiac);
cardiac_signal = single(comp.trial{1}(cardiac_idx, :));
fsample = comp.fsample;
t0 = comp.time{1}(1);
unmixing = comp.unmixing;
topo = comp.topo;
topolabel = comp.topolabel;
sk = [meas.sk]; ku = [meas.ku]; RR = [meas.RR]; Rampl = [meas.Rampl]; bpm = [meas.bpm];
RPeakstoNoise = [meas.RPeakstoNoise];
NotCardiac = double(vertcat(meas.NotCardiac));
thresh_sk = caracas_cfg.thresh_sk; thresh_ku = caracas_cfg.thresh_ku;
thresh_RR = caracas_cfg.thresh_RR; thresh_Rampl = caracas_cfg.thresh_Rampl;
thresh_bpm = caracas_cfg.thresh_bpm;

save(out_mat, 'cardiac_idx', 'cardiac_signal', 'fsample', 't0', 'unmixing', 'topo', 'topolabel', ...
    'sk', 'ku', 'RR', 'Rampl', 'bpm', 'RPeakstoNoise', 'NotCardiac', 'is_cardiac', ...
    'thresh_sk', 'thresh_ku', 'thresh_RR', 'thresh_Rampl', 'thresh_bpm', 'ncomp', 'rank_data', '-v7');
end

function data = preprocess_blockwise(data, blocksize)
nchan = numel(data.label);
refmean = mean(data.trial{1}, 1);
for b = 1:blocksize:nchan
    idx = b:min(b + blocksize - 1, nchan);
    data.trial{1}(idx, :) = data.trial{1}(idx, :) - refmean;
end
clear refmean
for b = 1:blocksize:nchan
    idx = b:min(b + blocksize - 1, nchan);
    cfg = [];
    cfg.channel = data.label(idx);
    cfg.hpfilter = 'yes';
    cfg.hpfreq = 1;
    cfg.hpfilttype = 'firws';
    tmp = ft_preprocessing(cfg, data);
    data.trial{1}(idx, :) = tmp.trial{1};
    clear tmp
end
end
