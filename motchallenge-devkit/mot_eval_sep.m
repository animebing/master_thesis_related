    function mot_eval_sep(det_type, res_dir)
% for one detection method defined by det_type, evaluate the result
% for fixed threshold
% DPM: 0.0
% SDP: 0.4
% FRCNN: 0.1

if strcmp(det_type, 'DPM')
    seqmap = 'c10-train-DPM.txt';
elseif strcmp(det_type, 'SDP')
    seqmap = 'c10-train-SDP.txt';
elseif strcmp(det_type, 'FRCNN')
    seqmap = 'c10-train-FRCNN.txt';
elseif strcmp(det_type, 'YOLO')
    seqmap = 'c10-train-YOLO.txt';
else
    error('unknown detection type %s', det_type)
end

seqmapFile=fullfile('seqmaps', seqmap);
assert(exist(seqmapFile,'file')>0,'seqmap file %s does not exist',seqmapFile);
fid = fopen(seqmapFile);
allseq = textscan(fid,'%s','HeaderLines',1);
fclose(fid);
allseq=allseq{1};
seq_len = length(allseq);

res_dir = [res_dir '/'];

res_mat = zeros(seq_len + 1, 17);
res_csv = fullfile(res_dir, ['result_' det_type '.csv']);

benchmarkDir = '../../MOT17/train/';

mat_file = fullfile(res_dir, ['allMets_' det_type '.mat']);

allMets = evaluateTracking(seqmap, res_dir, benchmarkDir, fid);

save(mat_file, 'allMets');
fclose(fid);
for i = 1:seq_len
    res_mat(i, :) = allMets.mets2d(i).m;
end

res_mat(end, :) = allMets.bmark2d;
csvwrite(res_csv, res_mat)
end

