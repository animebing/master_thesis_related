function write_to_csv(det_type, res_dir)    

if strcmp(det_type, 'DPM')
    seqmap = 'c10-train-DPM.txt';
    dir_idxs = 0:10;
elseif strcmp(det_type, 'SDP')
    seqmap = 'c10-train-SDP.txt';
    dir_idxs = 0:4;
elseif strcmp(det_type, 'FRCNN')
    seqmap = 'c10-train-FRCNN.txt';
    dir_idxs = 0:7;
else
    error('unknown detection type %s', det_type)
end

seqmapFile=fullfile('seqmaps',seqmap);
assert(exist(seqmapFile,'file')>0,'seqmap file %s does not exist',seqmapFile);
fid = fopen(seqmapFile);
allseq = textscan(fid,'%s','HeaderLines',1);
fclose(fid);
allseq=allseq{1};
seq_len = length(allseq);

res_dir = [res_dir '/' det_type '/'];

for i = 1:length(dir_idxs)
    res_mat = zeros(seq_len + 1, 17);
    dir_idx = dir_idxs(i);
    tmp_res_dir = [res_dir num2str(dir_idx) '/'];
    fprintf('processing: %s\n', tmp_res_dir);
    mat_file = [tmp_res_dir 'allMets.mat'];
    load(mat_file);
    for j = 1:seq_len
        res_mat(j, :) = allMets.mets2d(j).m;
    end
    res_mat(end, :) = allMets.bmark2d;
    res_csv = [res_dir 'result_' num2str(dir_idx) '.csv'];
    csvwrite(res_csv, res_mat)
    
end

end
