clear all
%close all
clc

%% IMPORT

opts = delimitedTextImportOptions("NumVariables", 7);
% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = " ";
% Specify column names and types
opts.VariableNames = ["Time_step", "Times", "Positionm", "Min", "Mean", "Max", "Std"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double"];
% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";
opts.LeadingDelimitersRule = "ignore";

% Import the data
%folder = 'C:\Users\fussegli\Desktop\Ubuntu_sharedfolder\AMREXcodes\echemAMR\models\CEAcharging\t_10\';
%folder = 'C:\Users\fussegli\Desktop\LIB_0D\t_100\';
folder = 'C:\Users\fussegli\Desktop\VBox_sharedfolder\AMREXcodes\echemAMR\models\CEAcharging\';
%folder = 'C:\Users\fussegli\Desktop\VBox_sharedfolder\AMREXcodes\echemAMR\models\';
anode_concentration = readtable([folder 'anode_concentration.csv'], opts);
anode_potential = readtable([folder 'anode_potential.csv'], opts);
cathode_concentration = readtable([folder 'cathode_concentration.csv'], opts);
cathode_potential = readtable([folder 'cathode_potential.csv'], opts);
electrolyte_concentration = readtable([folder 'electrolyte_concentration.csv'], opts);
electrolyte_potential = readtable([folder 'electrolyte_potential.csv'], opts);

% Clear temporary variables
clear opts

%% SELECT DATA

% Cs_a = anode_concentration.Mean;
% Phis_a = anode_potential.Mean;
% Cs_c = cathode_concentration.Mean;
% Phis_c = cathode_potential.Mean;
% Ce = electrolyte_concentration.Mean;
% Phie = electrolyte_potential.Mean;

[Cs_a_t, Cs_a] = cleandata(anode_concentration);
[Phis_a_t, Phis_a] = cleandata(anode_potential);
[Cs_c_t, Cs_c] = cleandata(cathode_concentration);
[Phis_c_t, Phis_c] = cleandata(cathode_potential);
[Ce_t, Ce] = cleandata(electrolyte_concentration);
[Phie_t, Phie] = cleandata(electrolyte_potential);

%% PLOT

Fig = figure;
ax_=axes(Fig);
hold(ax_,'on'); 
plot(Cs_a_t,Cs_a,"LineWidth",2,"LineStyle",'--',"DisplayName","Anode (AMREX)");
plot(Ce_t,Ce,"LineWidth",2,"LineStyle",'--',"DisplayName","Electrolyte (AMREX)");
plot(Cs_c_t,Cs_c,"LineWidth",2,"LineStyle",'--',"DisplayName","Cathode (AMREX)");
xlabel('Time (s)');
ylabel('Concentrations (mol.m^(-3))');
h_legend = legend(ax_,'Location','best');
hold(ax_,'off'); 

Fig = figure;
ax_=axes(Fig);
hold(ax_,'on'); 
plot(Phis_a_t,Phis_a,"LineWidth",2,"LineStyle",'--',"DisplayName","Anode (AMREX)");
plot(Phie_t,Phie,"LineWidth",2,"LineStyle",'--',"DisplayName","Electrolyte (AMREX)");
plot(Phis_c_t,Phis_c,"LineWidth",2,"LineStyle",'--',"DisplayName","Cathode (AMREX)");
xlabel('Time (s)');
ylabel('Potential (V)');
h_legend = legend(ax_,'Location','best');
hold(ax_,'off'); 

function [t,m] = cleandata(data)
    
    steps = data.Time_step;
    u_steps = unique(steps);
    n = length(u_steps);
    t = zeros(n,1);
    m = zeros(n,1);

    for k=1:1:n
        idx = find(steps==u_steps(k));
        t(k,1)=data.Times(idx(1));
        m(k,1) = mean(data.Mean(idx),'omitnan');
    end
end