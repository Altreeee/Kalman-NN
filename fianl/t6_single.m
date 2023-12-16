clc
clear

%% 初始化参数：CV 匀速运动
% 关于卡尔曼滤波与cv：https://blog.csdn.net/ouok000/article/details/125578636?spm=1001.2014.3001.5501
dt=1;       %雷达扫描周期=1s
T=80/dt;    %总采样次数=80次

%这些参数是根据公式推出来的
A=[1,dt,0,0; 
    0,1,0,0; 
    0,0,1,dt; 
    0,0,0,1];   %状态转移矩阵A
G=[0.5*dt^2,0; 
    dt,0; 
    0,0.5*dt^2; 
    0,dt]; %过程噪声驱动矩阵
H=[1,0,0,0; 
    0,0,1,0];   %观测矩阵

%改变过程噪声
a=1e-2;  %如果增大这个参数，目标真实轨迹会变成曲线 %1*10^（-2）
Q=a*eye(2); %过程噪声方差=a

%改变观测噪声
b=10;
R=b*eye(2); %观测噪声方差=b

%过程噪声与观测噪声均为高斯白噪声，均值为0，方差分别为Q和R
W=sqrtm(Q)*randn(2,T);  %过程噪声：产生均值为0，方差为Q的一个（2*T）随机数矩阵
V=sqrtm(R)*randn(2,T);  %观测噪声：高频随机干扰信号，均值为0，方差为R的一个（2*T）随机数矩阵

%% 初始化参数
X=zeros(4,T);   %目标真实值，速度。生成4xT全零阵
X(:,1)=[0,5,0,20];  %目标初始位置，速度。X的第一列的所有元素是一个列向量
%从100m,300m处出发，水平方向运动速度5m/s，垂直方向速度为2m/s

Z=zeros(2,T);   %传感器对应位置的观测
Z(:,1)=[X(1,1),X(3,1)]; %观测初始化[100,300]

Xkf=zeros(4,T); %kalman滤波状态初始化
Xkf(:,1)=X(:,1);

P=eye(4);   %协方差初始化

%% 模拟目标运动，观测站对目标进行观测，卡尔曼滤波，生成真实轨迹，观测轨迹和滤波轨迹

for k=2:T
    %对应2个状态方程
    %目标真实轨迹
    X(:,k)=A*X(:,k-1)+G*W(:,k-1);%系统的状态方程 W(:,k-1)是噪声数值、G是过程噪声驱动矩阵
    %目标观测轨迹
    Z(:,k)=H*X(:,k)+V(:,k);%系统的观测方程
    
    %对应卡尔曼滤波的5个方程
    %开始滤波
    Xpred=A*Xkf(:,k-1);     %第一步：状态预测
    Ppred=A*P*A'+G*Q*G';     %第二部：协方差预测
    K=Ppred*H'*inv(H*Ppred*H'+R);       %第三步：求增益，inv:矩阵求逆
    %{
    关于协方差矩阵：
        协方差矩阵的每个元素是原矩阵两两元素的协方差，
        因为是高斯白噪声，因此只有对角线有值，即自己与自己的协方差，
        而因为完全相同变量的协方差就是方差，所以这里的测量噪声的协方差矩阵直接使用R
    %}
    Xkf( : ,k)=Xpred+K*(Z( : ,k)-H*Xpred);      %第四步：状态更新
    P=(eye(4)-K*H)*Ppred;   %第五步：协方差更新
end

%% 绘制轨迹图
% 绘制轨迹图
figure;

% 绘制目标真实轨迹
subplot(2,1,1);
plot(X(1,:), X(3,:), 'b-', 'LineWidth', 2);  % X坐标 vs Y坐标
title('目标真实轨迹');
xlabel('X坐标');
ylabel('Y坐标');

% 绘制目标观测轨迹和卡尔曼滤波轨迹
subplot(2,1,2);
plot(Z(1,:), Z(2,:), 'bo-', 'LineWidth', 2);  % 观测X vs 观测Y
hold on;
plot(Xkf(1,:), Xkf(3,:), 'g-', 'LineWidth', 2);  % 滤波后的X vs 滤波后的Y
title('目标观测轨迹和卡尔曼滤波轨迹');
legend('观测轨迹', '卡尔曼滤波轨迹');
xlabel('X坐标');
ylabel('Y坐标');

% 设置整体标题
sgtitle('目标真实轨迹、观测轨迹和卡尔曼滤波轨迹');












