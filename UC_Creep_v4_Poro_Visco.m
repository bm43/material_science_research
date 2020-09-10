%% Unconfined Compression Creep Data Analysis Code
% Author: Axel Moore
% Date: 24 July 2019

%% File Open and Read
close all; clear all; clc;
[file,path] = uigetfile('*.csv');
disp(file);                             
delimiter = ',';
formatSpec = '%q%q%q%q%[^\n\r]';
fullFileName = fullfile(path, file);
[fileID, message] = fopen(fullFileName, 'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);

% close the text file
fclose(fileID);

% convert the contents of columns containing numeric text to numbers
% replace non-numeric text with NaN
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = dataArray{col};
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[1,2,3,4]
    % converts text in the input cell array to numbers, replaced non-numeric
    % text with NaN
    rawData = dataArray{col};
    for row=1:size(rawData, 1)
        % create a regular expression to detect and remove non-numeric prefixes and
        % suffixes
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData{row}, regexstr, 'names');
            numbers = result.numbers;
            
            % detected commas in non-thousand locations
            invalidThousandsSeparator = false;
            if any(numbers==',');
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(numbers, thousandsRegExp, 'once'));
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % convert numeric text to numbers
            if ~invalidThousandsSeparator;
                numbers = textscan(strrep(numbers, ',', ''), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch me
        end
    end
end

% replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw);   % find non-numeric cells
raw(R) = {NaN};                                         % replace non-numeric cells

% allocate imported array to column variable names
TimeElap = cell2mat(raw(:, 1)); % elaspsed time in seconds
TimeCyc = cell2mat(raw(:, 2));  % cycle time in seconds (resets at collection time)
DispOG = cell2mat(raw(:, 3));   % displacement of stage in mm
LoadOG = cell2mat(raw(:, 4));   % load in N

% clear temporary variables
clearvars raw R;

%% Invert Disp and Load; Set Contact Threshold for Fitting
prompt = {'sample diameter (mm):','sample thickness (mm):'};
title = 'Input';
dims = [1 35];
definput = {'6.0','1.0'};
answer = inputdlg(prompt,title,dims,definput);
D = str2num(answer{1,1});           % sample diameter (cylindrical speciment)
t = str2num(answer{2,1});           % sample thickness
ac = D/2;                           % sample radius
Disp = -(DispOG(:,1)-DispOG(5,1));  % zero disp and make positive
Load = -(LoadOG(:,1)-LoadOG(5,1));  % zero load and make positive

[k,l]=find(isnan(Disp))             % if this returns values other than 1-4 you need to find them 
                                    % and remove them from the data file

% find where load < 0.2 N (~sensor resolution)
[g,p] = find(Load > 0.2);           % g = coordinates of where load is > 0.2N

%% Zero Data with Threshold
% data set for fitting
[r,c] = size(g);
xdata = Disp(g(1,1):g(r,1),1);      % disp data set zeroed and with threshold
ydata = Load(g(1,1):g(r,1),1);      % load data set zeroed and with threshold
tdata = TimeElap(g(1,1):g(r,1),1);  % time data
mload = mean(ydata);                % average load/force

%% Find Approach Portion
[rr,cc] = size(ydata);
[u v] = find(ydata > 0.9*mload);                  % get coords of this value
xdataApp = xdata(1:u(1,1),1);                     % approch matrix (x)
ydataApp = ydata(1:u(1,1),1);                     % approch matrix (y) 

xdataStatic = xdata(u(1,1):rr-150,1);                   % relaxation data (x), -150 to remove unloading data
ydataStatic = ydata(u(1,1):rr-150,1);                   % relaxation data (y), -150 to remove unloading data
timeStatic = tdata(u(1,1):rr-150,1)-tdata(u(1,1),1);    % relaxation time (t), -150 to remove unloading data
S0 = mean(ydataStatic)/(ac^2*pi);                             % average applied stress (MPa)

%% Dectecting Contact and Initial Compressive Modulus
fun = @(d,xdataApp)(d(1)*(xdataApp-d(2))/t*(D/2)^2*pi); % function to fit: surface detection+compressive modulus
d0 = [0.5,0.2];                                         % initial guesses
[d,resnorm] = lsqcurvefit(fun,d0,xdataApp,ydataApp);    % optimization scheme

% fit values and equation
EcApp = d(1);                                           % approach modulus
DoffApp = d(2);                                         % surface offset (distance from sample)
RSSApp = resnorm;                           
TSSApp = sum((ydataApp(:,1)-mean(ydataApp)).^2);
R2App = 1-RSSApp/TSSApp;
FitApp =(EcApp*(xdataApp-DoffApp)/t*(D/2)^2*pi);        % model fit values   

%% Plot Results
figh = figure(1);             % plots a figure panel
plot(1:10,1:10)               % sets up the size and position of the panel fig
pos = get(figh,'position');
set(figh,'position',[pos(1:2)/4 pos(3:4)*2])

% load vs displacement (zeroed - dots)
% approach (open circles)
% approach fit (green line)
subplot (2,3,1)
plot(Disp,Load,'k.',xdataApp,ydataApp,'o',xdataApp,real(FitApp),'g-');
xlabel('displacement (mm)');
ylabel ('load (N)');
legend('Data','Approach Data','Fit');
xlim([0 inf]);
ylim([-inf,inf]);

% load versus time plot (diagnostic plot)
    % subplot (2,3,2)
    % plot(timeStatic,ydataStatic,'.');
    % xlabel('time (s)');
    % ylabel ('load (N)');

%% Modulus vs Time
% EcCorrect
EcCorrect = ydataStatic./((xdataStatic-DoffApp)/t*(D/2)^2*pi);  %substrate corrected modulus
xdataStaticC = xdataStatic-DoffApp;                             %corrected xdata for offset

%% Prep for Model Fitting
% strain and strain rate 
[row2,col2] = size(xdataStaticC);
j = 1;                                                                                          % number of cells to take average straina and strain rate over
for i = 1:row2-j;
    ep(i,1) = abs(xdataStaticC(i+j,1)/t);                                                       %strain
    ept(i,1) =((xdataStaticC(i+j,1)-xdataStaticC(i,1))/t)/(timeStatic(i+j,1)-timeStatic(i,1));  %strain rate
end

% adjust vector lengths
timeStatic1 = timeStatic(1:row2-j,1);           % this is to shift the time data for the shorter deformation rate matrix
EcCorrect1 = EcCorrect(1:row2-j,1);             % this is to shift the modulus data for the shorter deformation rate matrix
xdataStatic1 = xdataStaticC(1:row2-j,1);        % deformation data is shifted to vector size
equilmod = mean(EcCorrect1(end-1000:end,1));   % assumes that after 1000's the sample has approached an equilibrium condition and averages the values
Ff = (EcCorrect1-equilmod)./EcCorrect1;         % fluid load fraction calculated (subject to noise)

%% Logistic Function Fit to Strain Rate
% Used as a smooth function to fit to
fun4 = @(h,timeStatic1) (h(1)+(max(ept)-h(1))./(1+(timeStatic1./h(2)).^h(3)));
h0 = [1,1,1];

lbs = [0,0,0];              % lower bound of model
ubs = [inf,inf,inf];        % upper bound of model
[h,resnorm] = lsqcurvefit(fun4,h0,timeStatic1,ept,lbs,ubs); % optimization

A1 = max(ept);                                          % max value
A2 = h(1);                                              % min value
A3 = h(2);                                              % xc
A4 = h(3);                                              % power
logistic = A2+(A1-A2)./(1+(timeStatic1./A3).^A4);       % fit equation (fit data)

%% Plot Modulus vs Time
subplot (2,3,2)         
semilogx(timeStatic1,EcCorrect1,'k.');
xlabel('time (s)');
ylabel ('Modulus (MPa)');
legend('Data');
xlim([0.05 inf]);
ylim([-inf,inf]);

%% Resample Variables - Section Written by Gregor
% LOGDOWNSAMPLE Downsample x and y based on log(x + 1)
% The function calculates a step vector based on log(x + 1). It then uses
% this to loop through x and y, transcribing values into xd and yd.
% Inputs
% ======
% x - column vector of x values, e.g. time
% y - column vector of y values, e.g. load
%
% Outputs
% =======
% xd - column vector of downsampled x values
% yd - column vector of downsampled y values
x = timeStatic1;    % time
y = logistic;              % deformation rate  
z = Ff;             % fluid load fraction
w = ep;             % strain
length = size(y,1);

% Preallocating outputs for increased performance
xd = zeros(length, 1);  % time
yd = zeros(length, 1);  % deformation rate
zd = zeros(length, 1);  % fluid load fraction
wd = zeros(length, 1);  % strain
n = 1; % Input index
m = 1; % Output index

% 1+ -> So minimum index increment is at least 1
% (x+1) -> To limit log to positive values
% floor() -> Indices can only be integers (round() works too)
% Manipulate equation to bias sampling density as desired
% Raising to a power of 3 greatly reduces point density.
step = 1 + floor(log(x+1)).^3;
while n < length
    xd(m) = x(n);   % time
    yd(m) = y(n);   % deformation rate
    zd(m) = z(n);   % fluid load fraction
    wd(m) = w(n);   % strain
    n = n + step(n);
    m = m + 1;
end

% Trim outputs to last data point (m-1), getting rid of excess zeros
xd = xd(1:m-1); % time
yd = abs(yd(1:m-1)); % deformation rate
zd = zd(1:m-1); % fluid load fraction
wd = wd(1:m-1); % strain

%% Poroelastic Model
% poroelastic model with 2 floating variables (written to solve for F'(t))
fun2 = @(a,yd) (pi*ac^4*yd)/(8*a(2))...
    ./((equilmod*wd*pi*ac^2)+(pi*ac^4*yd)/(8*a(2)))... % fit to the data
    *(a(1)/(a(1)+2));                                       % equation 
a0 = [2,0.001];                                             % initial guesses 
lb = [0.01,0];                                              % lower bound of model
ub = [inf,inf];                                             % upper bound of model
[a,resnorm,residual] = lsqcurvefit(fun2,a0,yd,zd,lb,ub);    % output fit performance

% fit values and equation from data
E0 = equilmod;      % equilibrium modulus (MPa)
Estar = a(1);       % tensile modulus (MPa) E'
k = a(2);           % permeability (mm^4/Ns)
Ey = Estar*E0;      % tensile modulus (MPa) Etensile
%E0 is equilmod
% poroelastic model fit
FitPoro = (pi*ac^4*yd)./(8*k)...
    ./((E0*wd*pi*ac^2)+(pi*ac^4*yd)./(8*k))...%actual equation
    *((Ey/E0)/((Ey/E0)+2));

RSSPore = resnorm;                                      % residual sums squared wrt to the fit function
Resid = residual;                                       % residual wrt to the fit function
TSSPore = sum((zd(:,1)-mean(zd)).^2);                   % total sums squared wrt to the fit function
R2Pore = 1-RSSPore/TSSPore;                             % coefficient of determination wrt to the fit function

% plot results for model fit
subplot (2,3,4)
semilogx(x,z,'k.',xd,FitPoro,'r-',xd,Resid,'g-');
xlabel('time (s)');
ylabel('fluid load fraction');
xlim([0.05 inf]);
ylim([-0.1,inf]);
legend('Raw','Fit', 'Resid');

%% Poroelastic Model
Fi =(max(EcCorrect1)-E0)/max(EcCorrect1);                 % initial fluid load fraction
Estarpin = -(2*Fi)/(Fi-1);              % fix E*
Eypin = Estarpin*E0;                    % solve for fixed Ey

% poroelastic model with 1 floating variable (written to solve for F'(t))
fun3 = @(m,yd) (pi*ac^4*yd)/(8*m(1))...
    ./((equilmod*wd*pi*ac^2)+(pi*ac^4*yd)/(8*m(1)))...
    *(Estarpin/(Estarpin+2));           % equation to fit

m0 = k;                             % initial guess
lbb = 0;                                % lower bound of model
ubb = inf;                              % upper bound of model

[m,resnorm,residual] = lsqcurvefit(fun3,m0,yd,zd,lbb,ubb); % output fit performance
%zd=fluid load fraction
%yd=deformation rate
% fit values and equation
kpin = m(1);                            % permeability (mm^4/Ns) that is fitted


% poroelastic model fit
FitPoroPin = (pi*ac^4*yd)./(8*kpin)...
    ./((E0*wd*pi*ac^2)+(pi*ac^4*yd)./(8*kpin))...
    *((Estarpin)/((Estarpin)+2));

RSSPorePin = resnorm;                                           % residual sums squared wrt to the fit function
ResidPin = residual;                                            % residual wrt to the fit function
TSSPorePin = sum((zd(:,1)-mean(zd)).^2);                        % total sums squared wrt to the fit function
R2PorePin = 1-RSSPorePin/TSSPorePin;                            % coefficient of determination wrt to the fit function

%%
%parameters:
R=3; % in mm
k=a(2); % permeability in mm^4/Ns
area=2*pi*R*(wd+R);%total flow area?

coeff=Estar/(Estar+2);

Fe=((area*equilmod).*wd);

Fp=-(pi*yd/k).*( (log(wd+ac).*wd.^3)/2-(log(wd).*(wd.^3))/2-(ac*wd.^2)/2+(ac^2*wd)/4-(ac^3)/6);

Fprime=coeff.*Fp./(Fp+equilmod.*wd*(R^2)*pi);


%Rf is from
figure(4)

%plot(xd,zd,'k.',xd,Fprime,'r-');

%plot(xd,Fp)
%xlabel('strain')
%ylabel('fluid load fraction')
legend('Fp','zd');

%%
figure
plot(xd,zd,'k.',xd,FitPoro,'r-',xd,FitPoroPin,'g-');
%zd=fluid load fraction
%fitporo is equation
%fitporopin as well
xlabel('time (s)');
ylabel('fluid load fraction');
xlim([0.5 inf]);
ylim([-0.1,inf]);
legend('Raw','Fit (2 para)', 'Fit (1 para)');
%%


