function caracas_run(in_mat, out_mat)
d = load(in_mat);
n = size(d.src, 1);
comp = [];
comp.fsample = d.fs;
comp.trial = {d.src};
comp.time = {(0:size(d.src, 2) - 1) / d.fs};
comp.label = arrayfun(@(i) sprintf('IC%03d', i - 1), 1:n, 'uni', 0)';
[heart_IC, meas] = CARACAS(struct(), comp);
heart_IC = heart_IC(:)' - 1;
bpm = [meas.bpm];
sk = [meas.sk];
ku = [meas.ku];
rr_cv = [meas.RR];
rampl_cv = [meas.Rampl];
save(out_mat, 'heart_IC', 'bpm', 'sk', 'ku', 'rr_cv', 'rampl_cv');
end
