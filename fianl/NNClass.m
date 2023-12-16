%% NNClass: Nearest Neighbor Classification
%使用先验量测估计Z_f对未知分类的测量点Z分类


function [Z_out] = NNClass(Z_f, Z, TargetNum, S)
    
    
    % 初始化
    Z_out = cell(1, TargetNum);
    decided=zeros(2,TargetNum);
    undecided=zeros(2,TargetNum);
    
    
    for j=1:TargetNum
        % 取出Z_f每个cell的最后1项，生成decided矩阵
        decided(:,j) = Z_f{j}(:, end);
        % 初始化结果矩阵
        Z_out{1,j} = zeros(2,1);
        % 取出Z每个cell的最后1项，生成undecided矩阵
        undecided(:,j) = Z{j}(:, end);
    end
    
        
    % 对每个未知分类测量点进行分类
    for i = 1:TargetNum   
        distances=zeros(1,size(undecided,2));%size(undecided, 2) 表示获取矩阵 undecided 的列数
        for j = 1:size(undecided,2)
            % 计算马氏距离
            %distances(1,j) = sqrt(sum((decided(:,i) - undecided(:, j)).^2, 1));    
            distances(1,j) = ((undecided(:,j)-decided(:,i))')*inv(S)*(undecided(:,j)-decided(:,i));
           
        end
        
        % 找到距离最小的索引
        [~, minIndex] = min(distances);
        
        % 将对应的未知分类测量点放入XKnownout中
        Z_out{1,i} = undecided(:, minIndex);
        
        % 从undecided矩阵中删除已分类的测量点
        undecided(:, minIndex) = [];
    end
  
    
end



