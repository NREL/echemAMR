clear all
close all
clc

%% SOURCES
% [1] J. Laurencin, V. Roche, C. Jaboutian, I. Kieffer, J. Mougin, M.C. Steil, Ni-8YSZ cermet re-oxidation of anode supported solid oxide fuel cell: From kinetics measurements to mechanical damage prediction, International Journal of Hydrogen Energy, 2012, 37 (17), pp. 12557-12573.
% [2] C.H. Hsueh, A.G. Evans, Residual Stresses in Metal/Ceramic Bonded Strips, Journal of American Cermamic Society, 1985, 68 (5), pp. 241-248.
% [3] C.H. Hsueh, Thermal stresses in elastic multilayer systems, Thin Solid Films, 2002, 418, pp. 182-188.
% [4] C.-H. Hsueh, Modeling of elastic deformation of multilayers due to residual stresses and external bending, Journal of Applied Physics, 2002, 91 (12), pp. 9652- 9656.
% [5] C. H. Hsueh, S. Lee, T. J. Chuang, An Alternative Method of Solving Multilayer Bending Problems, Journal of Applied Mechanics, 2003, 70 (1), pp. 151-154.

% And for detailed derivation of the equations:
% https://tel.archives-ouvertes.fr/tel-01223428
% p229-240

%% MATERIAL COEFFICIENTS
e = [10e-6 20e-6]; % [m]
E = [10e9 20e9]; % [Pa]
mu = [0.3 0.3]; % Poisson ratio
a = [10e-6 20e-6]; % Coefficient of thermal expansion [K^-1]

%% LOADINGS
DeltaT = 10; % [K]

%% AMREX CASE
% e = [1 1]; % [m]
% E = [138.73 1.78]; % [Pa]
% mu = [0.3 0.3]; % Poisson ratio
% 
% beta = [0.1 2.0];
% deltaC = [0.1 0.1];
% 
% a=beta;
% DeltaT=deltaC;


%% GRID
ngrid=1000;

%%
%% MODEL
%%

% Effective E
Eeff = E./(1-mu);

% weight
p = (Eeff.*e)./(sum(Eeff.*e));

% Thermal strain
eth = a.*DeltaT;

% Total strain
etot = sum(eth.*p); 
% Rule of mixture weightd with product Yound modulus * layer thickness
% That is the layers with large Yound modulus * layer thickness control the total elastic strain

%% ELASTIC STRAIN W/O FLEXION
% Elastic strain (no flexion)
eel_plan = etot - eth;

%% ELASTIC STRAIN DUE TO FLEXION
nlayers = length(e);
y = zeros(1,nlayers+1);
for layer=1:1:nlayers
    y(layer+1) = y(layer)+e(layer);
end

% Neutral axis
num=0;
for i=1:1:nlayers
    num = num + ( Eeff(i) * (y(i)^2 - y(i+1)^2)/2 );
end
neutral_axis = - num/(sum(Eeff.*e));

% Curvature radius
num=0;
den=0;
for i=1:1:nlayers
    num = num + ( Eeff(i) * (neutral_axis*(y(i+1)^2-y(i)^2)/2 - (y(i+1)^3-y(i)^3)/3 ));
    den = den + ( Eeff(i) * eel_plan(i) * (y(i+1)^2-y(i)^2)/2 );
end
r= - num/den;

% Elastic strain due to flexion
eel_flexion=zeros(ngrid,1);
yy=linspace(0,y(end),ngrid);
for k=1:1:ngrid
    eel_flexion(k)= (neutral_axis-yy(k))/r;
end

%% STRESS
stress_noflexion=zeros(ngrid,nlayers)+NaN;
stress_flexion=zeros(ngrid,nlayers)+NaN;
stress_tot=zeros(ngrid,nlayers)+NaN;
layer=1;
for k=1:1:ngrid
    if yy(k)>y(layer+1)
        layer=layer+1;
    end
    stress_noflexion(k,layer) = Eeff(layer) * eel_plan(layer);
    stress_flexion(k,layer) = Eeff(layer) * eel_flexion(k);
    stress_tot(k,layer) = Eeff(layer) * ( eel_plan(layer) + eel_flexion(k) );
end

%% FIGURE
Fig = figure; % Create figure
Fig.Name= 'Elastic multilayer'; % Figure name
Fig.Color='white'; % Background colour
scrsz = get(0,'ScreenSize'); % Screen resolution
set(Fig,'position',[scrsz(1) scrsz(2) scrsz(3)*4/5 scrsz(4)/2]); % Full screen figure

smin = min([nanmin(nanmin(stress_noflexion)) nanmin(nanmin(stress_flexion)) nanmin(nanmin(stress_tot)) ]);
smax = max([nanmax(nanmax(stress_noflexion)) nanmax(nanmax(stress_flexion)) nanmax(nanmax(stress_tot)) ]);

for id_axe = 1:1:3
    clear str_legend
    sub_axes=subplot(1,3,id_axe,'Parent',Fig);
    hold(sub_axes,'on'); % Active subplot
    if id_axe==1
        h_title=title ('Stress w/o flexion \sigma^{el plan}'); % Set title font
        for i=1:1:nlayers
            h_=plot(stress_noflexion(:,i),yy,'LineWidth',2);
        end        
    elseif id_axe==2
        h_title=title ('Stress due to flexion \sigma^{el flexion}'); % Set title font
        for i=1:1:nlayers
            h_=plot(stress_flexion(:,i),yy,'LineWidth',2);
        end        
        plot([smin smax],[neutral_axis neutral_axis],'LineWidth',2,'Color',[0.5 0.5 0.5],'LineStyle','-.');
        
    else
        h_title=title ('Total stress \sigma^{el}'); % Set title font
        for i=1:1:nlayers
            h_=plot(stress_tot(:,i),yy,'LineWidth',2);
        end
    end

    for i=1:1:nlayers
        str_legend(i).name = ['Layer #' num2str(i)];
    end
    if id_axe==2 
        str_legend(i+1).name = 'Neutral axis';
    end

    for i=1:1:nlayers-1
        plot([smin smax],[y(i+1) y(i+1)],'LineWidth',2,'Color','k','LineStyle','--');
    end

    plot([0 0],[min(y) max(y)],'LineWidth',2,'Color','k','LineStyle',':');
    xlabel('In-plane Stress \sigma_{xx}=\sigma_{zz} (Pa) ');
    ylabel('Thickness y (m)');
    grid(sub_axes,'on'); % Display grid
    legend(sub_axes,str_legend.name,'Location','best');
    set(sub_axes,'FontName','Times new roman','FontSize',14); % Fontname and fontsize
    h_title.FontSize = 16; % Set title fontsize
    hold(sub_axes,'off'); % Relase figure
end
sgtitle(Fig,'Stress profile for an elastic multilayer with uniform loading (delta T) and heterogeneous CTE','FontWeight','bold','FontSize',19,'FontName','Times new roman');


