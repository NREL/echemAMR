clear all
%close all
clc

%% IMPORT
folder = 'C:\Users\fussegli\Desktop\VBox_sharedfolder\AMREXcodes\echemAMR\models\CEAcharging\';

opts = delimitedTextImportOptions("NumVariables", 5);
% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = ",";
% Specify column names and types
opts.VariableNames = ["SOC", "Ds", "Ks", "Io", "OCP"];
opts.VariableTypes = ["double", "double", "double", "double", "double"];
% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";
opts.LeadingDelimitersRule = "ignore";
% Import the data
anode_coefficients = readtable([folder 'Anode_coefficients.csv'], opts);
cathode_coefficients = readtable([folder 'Cathode_coefficients.csv'], opts);

opts = delimitedTextImportOptions("NumVariables", 6);
% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = ",";
% Specify column names and types
opts.VariableNames = ["Ce", "De", "Ke", "tp", "Ac", "Kd"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double"];
% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";
opts.LeadingDelimitersRule = "ignore";
% Import the data
electrolyte_coefficients = readtable([folder 'Electrolyte_coefficients.csv'], opts);

%% PLOT
scrsz = get(0,'ScreenSize'); % Screen resolution
Fig = figure; % Create figure
Fig.Name= 'Cathode'; % Figure name
Fig.Color='white'; % Background colour
set(Fig,'position',[scrsz(1) scrsz(2) scrsz(3)*1 scrsz(4)*1]); % Full screen figure
for id_axe=1:1:4
    if id_axe==1
        sub_axes=subplot(2,2,id_axe,'Parent',Fig); hold(sub_axes,'on'); % Active subplot
        h_title=title ('Diffusivity');
        plot(cathode_coefficients.SOC,cathode_coefficients.Ds,'Linewidth',2);
        ylabel('Ds [m.s^{-2}]');
    elseif id_axe==2
        sub_axes=subplot(2,2,id_axe,'Parent',Fig); hold(sub_axes,'on'); % Active subplot
        h_title=title ('Conductivity');
        plot(cathode_coefficients.SOC,cathode_coefficients.Ks,'Linewidth',2);
        ylabel('Ds [S.m^{-1}]');
    elseif id_axe==3
        sub_axes=subplot(2,2,id_axe,'Parent',Fig); hold(sub_axes,'on'); % Active subplot
        h_title=title ('Exchange current density');
        plot(cathode_coefficients.SOC,cathode_coefficients.Io,'Linewidth',2);
        ylabel('Io [A.m^{-2}]');
    elseif id_axe==4
        sub_axes=subplot(2,2,id_axe,'Parent',Fig); hold(sub_axes,'on'); % Active subplot
        h_title=title ('Open circuit voltage');
        plot(cathode_coefficients.SOC,cathode_coefficients.OCP,'Linewidth',2);
        ylabel('OCP [V]');        
    end
    xlabel('SOC');
    % - Grid
    grid(sub_axes,'on'); % Display grid
    set(sub_axes,'XMinorGrid','on','YMinorGrid','on'); % Display grid for minor thicks
    set(sub_axes,'FontName','Times new roman','FontSize',12); % Fontname and fontsize
    h_title.FontSize = 14; % Set title fontsize
    h_legend.FontSize = 12; % Set title fontsize
    hold(sub_axes,'off'); % Relase figure
end
sgtitle('Cathode','FontName','Times new roman','FontSize',16);

Fig = figure; % Create figure
Fig.Name= 'Anode'; % Figure name
Fig.Color='white'; % Background colour
set(Fig,'position',[scrsz(1) scrsz(2) scrsz(3)*1 scrsz(4)*1]); % Full screen figure
for id_axe=1:1:4
    if id_axe==1
        sub_axes=subplot(2,2,id_axe,'Parent',Fig); hold(sub_axes,'on'); % Active subplot
        h_title=title ('Diffusivity');
        plot(anode_coefficients.SOC,anode_coefficients.Ds,'Linewidth',2);
        ylabel('Ds [m.s^{-2}]');
    elseif id_axe==2
        sub_axes=subplot(2,2,id_axe,'Parent',Fig); hold(sub_axes,'on'); % Active subplot
        h_title=title ('Conductivity');
        plot(anode_coefficients.SOC,anode_coefficients.Ks,'Linewidth',2);
        ylabel('Ds [S.m^{-1}]');
    elseif id_axe==3
        sub_axes=subplot(2,2,id_axe,'Parent',Fig); hold(sub_axes,'on'); % Active subplot
        h_title=title ('Exchange current density');
        plot(anode_coefficients.SOC,anode_coefficients.Io,'Linewidth',2);
        ylabel('Io [A.m^{-2}]');
    elseif id_axe==4
        sub_axes=subplot(2,2,id_axe,'Parent',Fig); hold(sub_axes,'on'); % Active subplot
        h_title=title ('Open circuit voltage');
        plot(anode_coefficients.SOC,anode_coefficients.OCP,'Linewidth',2);
        ylabel('OCP [V]');        
    end
    xlabel('SOC');
    % - Grid
    grid(sub_axes,'on'); % Display grid
    set(sub_axes,'XMinorGrid','on','YMinorGrid','on'); % Display grid for minor thicks
    set(sub_axes,'FontName','Times new roman','FontSize',12); % Fontname and fontsize
    h_title.FontSize = 14; % Set title fontsize
    h_legend.FontSize = 12; % Set title fontsize
    hold(sub_axes,'off'); % Relase figure
end
sgtitle('Anode','FontName','Times new roman','FontSize',16);

Fig = figure; % Create figure
Fig.Name= 'Electrolyte'; % Figure name
Fig.Color='white'; % Background colour
set(Fig,'position',[scrsz(1) scrsz(2) scrsz(3)*1 scrsz(4)*1]); % Full screen figure
for id_axe=1:1:5
    if id_axe==1
        sub_axes=subplot(2,3,id_axe,'Parent',Fig); hold(sub_axes,'on'); % Active subplot
        h_title=title ('Diffusivity');
        plot(electrolyte_coefficients.Ce,electrolyte_coefficients.De,'Linewidth',2);
        ylabel('Ds [m.s^{-2}]');
    elseif id_axe==2
        sub_axes=subplot(2,3,id_axe,'Parent',Fig); hold(sub_axes,'on'); % Active subplot
        h_title=title ('Conductivity');
        plot(electrolyte_coefficients.Ce,electrolyte_coefficients.Ke,'Linewidth',2);
        ylabel('Ds [S.m^{-1}]');
    elseif id_axe==3
        sub_axes=subplot(2,3,id_axe,'Parent',Fig); hold(sub_axes,'on'); % Active subplot
        h_title=title ('Transference number');
        plot(electrolyte_coefficients.Ce,electrolyte_coefficients.tp,'Linewidth',2);
        ylabel('t+ []');
    elseif id_axe==4
        sub_axes=subplot(2,3,id_axe,'Parent',Fig); hold(sub_axes,'on'); % Active subplot
        h_title=title ('Activity coefficient');
        plot(electrolyte_coefficients.Ce,electrolyte_coefficients.Ac,'Linewidth',2);
        ylabel('Ac []');     
    elseif id_axe==5
        sub_axes=subplot(2,3,id_axe,'Parent',Fig); hold(sub_axes,'on'); % Active subplot
        h_title=title ('Diffusional conductivity');
        plot(electrolyte_coefficients.Ce,electrolyte_coefficients.Kd,'Linewidth',2);
        ylabel('Kd [C.m^{-1}.s^{-1}]');           
    end
    xlabel('Ce [mol.m^{-3}]');
    % - Grid
    grid(sub_axes,'on'); % Display grid
    set(sub_axes,'XMinorGrid','on','YMinorGrid','on'); % Display grid for minor thicks
    set(sub_axes,'FontName','Times new roman','FontSize',12); % Fontname and fontsize
    h_title.FontSize = 14; % Set title fontsize
    h_legend.FontSize = 12; % Set title fontsize
    hold(sub_axes,'off'); % Relase figure
end
sgtitle('Electrolyte','FontName','Times new roman','FontSize',16);


