function mot_eval_det(det_type, res_dir)
% for one detection method defined by det_type, evaluate the result
% for different threshold
% DPM: [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
% SDP: [0.4, 0.5, 0.6, 0.7, 0.8]
% FRCNN: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

if strcmp(det_type, 'DPM')
    seqmap = 'c10-train-DPM.txt';
    dir_idxs = 0:10;
elseif strcmp(det_type, 'SDP')
    seqmap = 'c10-train-SDP.txt';
    dir_idxs = 0:4;
elseif strcmp(det_type, 'FRCNN')
    seqmap = 'c10-train-FRCNN.txt';
    dir_idxs = 0:7;
elseif strcmp(det_type, 'YOLO')
    seqmap = 'c10-train-YOLO.txt';
    dir_idxs = 0:7;
elseif strcmp(det_type, 'YOLO_new')
    seqmap = 'c10-train-YOLO.txt';
    dir_idxs = 0:7;
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

benchmarkDir = '../../MOT17/train/';
res_dir = [res_dir '/' det_type '/'];
res_txt = fullfile(res_dir, 'prints_all.txt');
fid = fopen(res_txt, 'w');

for i = 1:length(dir_idxs)
    dir_idx = dir_idxs(i);
    tmp_res_dir = [res_dir num2str(dir_idx) '/'];
    tmp_res_mat = zeros(seq_len + 1, 17);
    tmp_res_csv = fullfile(res_dir, ['result_' num2str(i) '.csv']);
    tmp_mat_file = fullfile(tmp_res_dir, ['allMets_' num2str(i) '.mat']);
    
    allMets = evaluateTracking(seqmap, tmp_res_dir, benchmarkDir, fid);
    save(tmp_mat_file, 'allMets');
    for j = 1:seq_len
        tmp_res_mat(j, :) = allMets.mets2d(j).m;
    end
    
    tmp_res_mat(end, :) = allMets.bmark2d;
    csvwrite(tmp_res_csv, tmp_res_mat);
end

fclose(fid);

end
