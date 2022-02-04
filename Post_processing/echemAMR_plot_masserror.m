clear all
%close all
clc

%% IMPORT
opts = delimitedTextImportOptions("NumVariables", 5);
% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = " ";
% Specify column names and types
opts.VariableNames = ["Time_step", "Times", "Anode", "Electrolyte", "Cathode"];
opts.VariableTypes = ["double", "double", "double", "double", "double"];
% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";
opts.LeadingDelimitersRule = "ignore";

% Import the data
folder = 'C:\Users\fussegli\Desktop\VBox_sharedfolder\AMREXcodes\echemAMR\models\CEAcharging\';
mass_error = readtable([folder 'Masserror.csv'], opts);

%% PLOT
Fig = figure;
Fig.Name= 'Mass conservation error'; % Figure name
Fig.Color='white'; % Background colour
ax_ = axes('parent',Fig);
hold(ax_,'on'); 
h_title=title ('Mass conservation error'); % Set title font
plot(mass_error.Times,mass_error.Anode,"LineWidth",2,"LineStyle",'-',"DisplayName","Anode");
plot(mass_error.Times,mass_error.Electrolyte,"LineWidth",2,"LineStyle",'-',"DisplayName","Electrolyte");
plot(mass_error.Times,mass_error.Cathode,"LineWidth",2,"LineStyle",'-',"DisplayName","Cathode");
xlabel('Time (s)');
ylabel('Mass error (%)');
h_legend = legend(ax_,'Location','best');
grid(ax_,'on'); % Display grid
set(ax_,'FontName','Times new roman','FontSize',14); % Fontname and fontsize
h_title.FontSize = 16; % Set title fontsize
hold(ax_,'off'); % Relase figure

