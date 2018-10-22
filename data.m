
dp1={'part_A','part_B'};
dp2={'test_data','train_data'};
for i=1:2
    for j=1:2
        path=['data\ShanghaiTech\',dp1{1,i},'\',dp2{1,j},'\ground-truth\']
        files=dir([path,'*.mat']);
        size0 = size(files);
        length = size0(1);
        for k=1:length
            files(k).name;
            load([path,files(k).name]);
            number=image_info{1,1}.number;
            points=image_info{1,1}.location;
            csvFileName=[files(k).name(1:end-4),'.csv'];
            fid=fopen([path,csvFileName],'w');
            fprintf(fid,'%d\n',number);
            size(points);
            for s=1:size(points,1)
                fprintf(fid,'%f,%f\n',points(s,1),points(s,2));
            end
            fclose(fid);
        end
    end
end

