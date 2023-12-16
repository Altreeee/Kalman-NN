clear all;
close all;

%%
%最近邻法NN+卡尔曼滤波KF 实现多目标跟踪
%单个观测站，n个目标，模拟单站多目标跟踪过程

%% 初始化参数
T=10;   %仿真时间长度

% 1，改变目标个数
TargetNum=3;    %目标个数
dt=1;   %采样时间间隔

%采用CV模型
F=[1, dt, 0, 0;
    0, 1, 0, 0;
    0, 0, 1, dt;
    0, 0, 0, 1];    %状态转移矩阵
G=[0.5*dt^2,0; 
    dt,0; 
    0,0.5*dt^2; 
    0,dt];   %过程噪声驱动矩阵
H=[1,0,0,0; 
    0,0,1,0];   %观测矩阵

%2，改变过程噪声
a=1e-2;  
Q=a*eye(2); %过程噪声方差=a

%3，改变观测噪声
b=10;
R=b*eye(2); %观测噪声方差=b

%% 【选A】状态和观测的初始化
%{
 for i=1:TargetNum
      
     %4，改变平行轨迹之间的间隔，对比分类效果：20/100
     d=100;
     %d=20;
     
     %状态初始化：目标初始位置，速度。
     %y的位置和y方向的速度是随机的
     X{i}(:,1)=[3,100/T,d*(i-1),100/T+0.1*randn];   %X中第i个cell的第一列 [x, vx, y, vy]
     
     %观测初始化：传感器对位置的观测
     Z{i}(:,1)=H*X{i}(:,1)+sqrt(R)*randn(2,1);  %产生均值为0，方差为R的一个 观测噪声
     
     Xkf{i}=zeros(4,T);
     Xkf{i}(:,1)=X{i}(:,1);     %卡尔曼滤波初始化
     
     P=eye(4);
     
 end
%}
 
 %% 【选B】调整轨迹方向，使之相交

 d=60;
 
 %状态初始化
 X{1}(:,1)=[10,100/T,80,-100/T+0.1*randn];
 X{2}(:,1)=[0,100/T,0,50/T+0.1*randn];
 X{3}(:,1)=[50, 0, 10, 50/T+0.1*randn];
 
 %观测初始化
 for i=1:TargetNum
     Z{i}(:,1)=H*X{i}(:,1)+sqrt(R)*randn(2,1);
     Xkf{i}=zeros(4,T);
     Xkf{i}(:,1)=X{i}(:,1);
     P=eye(4);
 end
 
%}
for j=1:TargetNum
    Xknown{1,j}=Xkf{1,j}(:,1);      %初始化已知的样本集合 
    Z_final{1,j}=Z{1,j}(:,1);
    %Xkf{1,j}(:,1)：Xkf的第j个cell的第1列
    %Xknown的每j个cell是Xkf的第j个cell的第一列，即为3个目标的初始值
end
 
 %% 模拟目标运动，观测站对目标进行观测以及卡尔曼滤波
 
 for t=2:T
     for j=1:TargetNum
         
         %第j个目标的状态方程
         X{j}(:,t)=F*X{j}(:,t-1)+G*sqrt(Q)*randn(2,1);  
         %单个观测站对第j个目标进行观测，观测方程【虽然Z中按目标分了cell，但实际上并不能知道每个点来自哪个目标】
         Z{j}(:,t)=H*X{j}(:,t)+sqrt(R)*randn(2,1);
         
         
         %卡尔曼滤波时间更新【预测】
         Xpred{j}=F*Xkf{j}(:,t-1);     %第一步：状态预测
         Ppred{j}=F*P*F'+G*Q*G';     %第二部：协方差预测
     end
     
     for j=1:TargetNum
         
         %S(k)=H*Ppred*H'+R【用来计算马氏距离】
         S=H*Ppred{j}*H'+R;
         
         %先验量测估计【猜测本次应该观察到的点位（注意Z代表实际观测到的点位）】
         Z_f{j}(:,t)=H*Xpred{j};
     end
     
     for j=1:TargetNum
         %测量Z_f中点与Z中点的马氏距离，进行数据关联分类，获得本次的已知分类测量点
         Z_out=NNClass(Z_f, Z, TargetNum, S);
     end
     for j=1:TargetNum
         %测量Z_f中点与Z中点的马氏距离，进行数据关联分类，获得本次的已知分类测量点
         Z_final{1,j}=[Z_final{1,j}, Z_out{1,j}];
     end
     
     for j=1:TargetNum
         %卡尔曼滤波状态更新【矫正】
         K=Ppred{j}*H'*inv(H*Ppred{j}*H'+R);       %第三步：求增益，inv:矩阵求逆
         Xkf{j}( : ,t)=Xpred{j}+K*(Z_final{j}( : ,t)-H*Xpred{j});      %第四步：状态更新
         P=(eye(4)-K*H)*Ppred{j};   %第五步：协方差更新
     end
 end
 
 %% 画图
figure
hold on,box on;
%title(['目标数='，num2str(TargetNum),', 初始轨迹间隔=',num2str(d),',过程噪声方差']);

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'];
for j=1:TargetNum
    color_index = mod(j, length(colors)) + 1; % 使用颜色数组轮转
    h1=plot(X{j}(1,:),X{j}(3,:),'-r.')  %多个目标的真实轨迹
    h2=plot(Z{j}(1,:),Z{j}(2,:),'r*')  %单个观测站对多个目标的观测轨迹
    h3=plot(Xkf{j}(1, :), Xkf{j}(3, :), ['-', colors(color_index), '.'], 'LineWidth', 1.5);
    h4=plot(Xkf{j}(1,:),Xkf{j}(3,:),'bo') %卡尔曼滤波后的轨迹点
end

legend([h1,h2,h3,h4],'真实轨迹','观测样本点','卡尔曼滤波样本轨迹,''卡尔曼滤波后的轨迹点')
%legend([h1,h2],'真实轨迹','观测样本点')
     
     


















