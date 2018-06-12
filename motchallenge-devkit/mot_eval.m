function mot_eval(res_dir)
% evaluate on mot17 training set    

benchmarkDir = '../../MOT17/train/';
seqmap = 'c10-train.txt';
seqmapFile=fullfile('seqmaps',seqmap);
assert(exist(seqmapFile,'file')>0,'seqmap file %s does not exist',seqmapFile);
fid = fopen(seqmapFile);
allseq = textscan(fid,'%s','HeaderLines',1);
fclose(fid);
allseq=allseq{1};
seq_len = length(allseq);

res_txt = fullfile(res_dir, 'all_prints.txt');
fid = fopen(res_txt, 'w');

allMets = evaluateTracking(seqmap, res_dir, benchmarkDir, fid);
fclose(fid);
mat_file = fullfile(res_dir, 'allMets.mat');
save(mat_file, 'allMets');

res_mat = zeros(seq_len + 1, 17);
for j = 1:seq_len
    res_mat(j, :) = allMets.mets2d(j).m;
end
res_mat(end, :) = allMets.bmark2d;
 
res_csv = fullfile(res_dir, 'results.csv');
csvwrite(res_csv, res_mat)
       
end