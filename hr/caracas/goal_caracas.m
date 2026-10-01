function n_done = goal_caracas(src_dir, out_dir, max_seconds)

setup_paths();
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
t_start = tic;
n_done = 0;
while toc(t_start) < max_seconds
    files = dir(fullfile(src_dir, '*.mat'));
    if isempty(files), break, end
    f = fullfile(src_dir, files(1).name);
    out_mat = fullfile(out_dir, files(1).name);
    t1 = tic;
    try
        score_file(f, out_mat);
    catch err
        fid = fopen(strrep(out_mat, '.mat', '.caracas_failed'), 'w');
        fprintf(fid, '%s', getReport(err, 'extended', 'hyperlinks', 'off'));
        fclose(fid);
    end
    delete(f);
    n_done = n_done + 1;
    fprintf('%s %.0fs\n', files(1).name, toc(t1));
end
end

function score_file(f, out_mat)
d = load(f);
src = double(d.src);
EEG = eeg_emptyset;
EEG.setname = 'goal';
EEG.nbchan = size(d.mixing, 1);
EEG.trials = 1;
EEG.pnts = size(src, 2);
EEG.srate = double(d.fs);
EEG.xmin = 0;
EEG.xmax = (EEG.pnts - 1) / EEG.srate;
EEG.times = (0:EEG.pnts - 1) / EEG.srate * 1000;
EEG.icaact = src;
EEG.icawinv = double(d.mixing);
EEG.icaweights = double(d.unmixing);
EEG.icasphere = eye(EEG.nbchan);
EEG.icachansind = 1:EEG.nbchan;
labels = cellstr(d.chlabel);
EEG.chanlocs = struct('labels', labels(:)', 'X', num2cell(d.chpos(:, 1))', ...
    'Y', num2cell(d.chpos(:, 2))', 'Z', num2cell(d.chpos(:, 3))');
EEG.chanlocs = convertlocs(EEG.chanlocs, 'cart2all');

cfg = SASICA('getdefs');
names = fieldnames(cfg);
for m = 1:numel(names)
    if isstruct(cfg.(names{m})) && isfield(cfg.(names{m}), 'enable')
        cfg.(names{m}).enable = false;
    end
end
cfg.CARACAS.enable = true;
cfg.opts.noplot = 1;
cfg.opts.noplotselectcomps = 1;
EEG = eeg_SASICA(EEG, cfg);

meas = EEG.reject.SASICA.icaCARACAS;
ccfg = meas(1).cfg;
meas = rmfield(meas, 'cfg');
is_cardiac = EEG.reject.SASICA.icarejCARACAS;
sk = [meas.sk]; ku = [meas.ku]; RR = [meas.RR]; Rampl = [meas.Rampl]; bpm = [meas.bpm];
RPeakstoNoise = [meas.RPeakstoNoise];
NotCardiac = double(vertcat(meas.NotCardiac));
thresh = [ccfg.thresh_sk, ccfg.thresh_ku, ccfg.thresh_RR, ccfg.thresh_Rampl, ccfg.thresh_bpm];
save(out_mat, 'is_cardiac', 'sk', 'ku', 'RR', 'Rampl', 'bpm', 'RPeakstoNoise', 'NotCardiac', 'thresh', '-v7');
end

function setup_paths()
persistent done
if ~isempty(done), return, end
home = getenv('HOME');
ftdir = fullfile(home, 'Documents', 'MATLAB', 'fieldtrip');
sasdir = fullfile(home, 'code', 'tools', 'SASICA');
here = fileparts(mfilename('fullpath'));
restoredefaultpath;
addpath(here);
addpath(ftdir);
ft_defaults;
addpath(sasdir);
addpath(fullfile(sasdir, 'CARACAS', 'heart_functions'));
addpath(fullfile(sasdir, 'eeglab'));
addpath(genpath(fullfile(sasdir, 'eeglab', 'functions')));
addpath(fullfile(home, 'code', 'tools', 'matlab_shim'));
ft_warning('off', 'FieldTrip:dataContainsNaN');
done = true;
end
