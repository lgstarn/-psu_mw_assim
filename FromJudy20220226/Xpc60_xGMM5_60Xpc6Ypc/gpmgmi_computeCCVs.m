%% Only observations from RTTOV_dc are deconvolved 
    % the user-defined options
    options = struct; 
    %%% A structure array is a data type that groups related data using data containers called fields. 
    %%% Each field can contain any type of data. Access data in a field using dot notation of the form structName.fieldName.
    %%% s = struct creates a scalar (1-by-1) structure with no fields.

    % --------------- Variables in Each Vector  ----------------------------
    % Names of geophysical variables to read from the model file
    options.model_vars = {'UV','W','T','P','Qvapor','QCloud','QRain','QIce','QSnow','QGraup'};%,'QHail'};
    % the names of the observation vars
    % SSMIS
    %options.obs_vars = {'19.35 V','22.24 V','37.00 V','91.66 H','150.00 H','183.31+/1 H','183.31+/6.6 H'};
    %options.obs_channels = [4,5,6,7,1,3,2];
    % GMI
    options.sensor = 'gpm_gmi';
    options.obs_vars = {'10.65V','10.65H','18.70V','18.70H','23.80V','36.50V','36.50H','89.00V','89.00H','166.0V','166.0H','183.31+/-3V','183.31+/-7V'};
    options.obs_channels = [1:13];

    if (length(options.obs_vars) ~= options.obs_channels)
        error('Mismatch in length between obs_vars and obs_channels')
    end

    % ----------------- Database Location -----------------------
    options.model_files = '/work2/06191/tg854905/stampede2/AllData/Pro1_CCVoperator/Harvey/DATABASE3/model/MW_THO/60_members/*';
    options.obs_directories = {'/work2/06191/tg854905/stampede2/AllData/Pro1_CCVoperator/Harvey/DATABASE3/simulated_BT/MW_THO/',}
    options.exps_all = {{'nodc'}, {'none'}};

    % ----------------- Result Path ---------------------------
    options.resultPath = '/work2/06191/tg854905/stampede2/AllData/Pro1_CCVoperator/Harvey/DATABASE3/Results/MW_THO/gpm_gmi_nodc/60Xpc_5xGMM_60Xpc6Ypc/'; 

    % ----------- Setup of Model-equivalent BT ------------------------ 
     % Which Rtm is used to calculate Tb
    if (1 == 1) 
        options.obs_CRTM = true;
        options.obs_RTTOV = false;
    else
        options.obs_CRTM = false;
        options.obs_RTTOV = true;
    end
     % If Convolution-Deconvolution is used
    if (1 == 1)
        options.deconvolution = false;
        options.use_geometry_file = false;
    elseif (1 == 2) 
        options.deconvolution = true;
        options.use_geometry_file = false;
    elseif (1 == 2)
        options.deconvolution = true;
        options.use_geometry_file = true;
        % the following options are only relevant if use_geometry_file is true
        options.obs_file_rep = '_dc.nc';
        options.geometry_file_rep = '_noise002.nc';
        options.model_lat_name = 'XLAT';
        options.model_lon_name = 'XLONG';
    end
                % ---------------------- %
    % ----------- Parameters of CCVs training  ------------------------     
                % ---------------------- %
    
    % -----------  Single column or multiple column -----------
     % Number of surrounding columns around a model-equivalent BT
    options.numSurCols = 0;

    % ---------------- Levels to Normalize ------------------
     % Whether to normalize by level
    if (1 == 1) % Part of variables normalized by level
        options.PartnormalizeByLevel = true;
    elseif (1 == 2) % Normalized by level
        options.PartnormalizeByLevel = false;
        options.normalizeByLevel = true;
    elseif (1 == 2) % None of variable normalized by level
        options.PartnormalizeByLevel = false;
        options.normalizeByLevel = false;
    end

    % --------------- Number of PCs/CCVs ------------------------
     % the minimum/maximum number of principal components to retain in the model and obs
    options.nmodelsvd_max = 60; %300*(2*options.numSurCols+1)^2;
    options.nmodelsvd_min = 60; %300*(2*options.numSurCols+1)^2;
    options.nobssvd_max = 6;  
    options.nobssvd_min = 6;
    
    % the maximum number of CCVs to plot/save
    options.maxCcvs = 6;

    % ------------ Land to Mask ---------------------------------
     % Whether to mask out land
    options.maskLand = true;
     % Read maskLand values from model or observation files
     % CRTM and RTTOV_rtm: options.maskLand_model = true; 
     % RTTOV_dc: options.maskLand_model = false;
    if (options.maskLand)
        if (1 == 1)
            options.maskLand_model = true; 
        else
            options.maskLand_model = false;
        end
    end
    
    % ----------- Global or Regional ---------------------------
     % computeGlobalStats: if true, compute statistics globally.  If false, use
     % the previously calculated global statistics to compute  
     % individual region statistics.
    if (1 == 2)
        options.computeGlobalStats = true;
    else
        options.computeGlobalStats = false;
    end

    % The number of regions to compute.                                                                       
    options.nregions = 5;

    % --------- Clustering Method ------------------------------
     % Gaussian Mixture Model or K-means 
    if (1 == 1)
        options.GMMs = true;
        options.kmeans = false;
    else
        options.GMMs = false;
        options.kmeans = true;
    end
    
     % Input Parameters to  GMMs
    if (options.GMMs) 
        options.GMMsnpc = 60;
        options.numGMM = options.nregions;
    end
        
    close all
    %-------------------- End User Definitions --------------------
    
   
             
    % number of boundary points to throw away in lat
    options.nlatr = 2;
    
    % number of boundary points to throw away in lon
    options.nlonr = 2;
    
    % number of initial scenes to disregard
    % options.nspinup = 1;
    options.nspinup = 0;
   
    % Result Path
    Statdir = options.resultPath;

    % get rid of the troublesome last newline character - probably doesn't
    % work on PC.  On a PC, use dir instead (not implemented).
    % disp([' options.model_dir']);
    options.model_dir = regexprep(ls(options.model_files),'\n$', '');
    % split the file list into individual strings
    options.model_dir = regexp(options.model_dir,'\n','split');
    %%% \n inserts a newline character
    %%% 'split' split a character vector into several substrings, where
    %%% each string is delimited by a \n character
    
    % ----------- Decide if the job is to calculate global statistics or not ------------
    % if it is, keep calculating
    % if it is not, read previously calculated global statistics to calculate regional statistics 
    if (options.computeGlobalStats) 
        % if computing the regioning, only use one global mean/stddev/cov
        options.nr = 1;
                
        if (options.kmeans)
            options.generators = {};
        end
        
    else
        % otherwise, compute the statistics for the regions.
        options.nr = options.nregions;
        
        options.model_mean_g = dlmread([Statdir,'gpm_gmi_global_mean_model.txt']);    
        options.obs_mean_g = dlmread([Statdir,'gpm_gmi_global_mean_obs.txt']);        
        options.model_stddev_g = dlmread([Statdir,'gpm_gmi_global_stddev_model.txt']);
        options.obs_stddev_g = dlmread([Statdir,'gpm_gmi_global_stddev_obs.txt']);    
        
        if (options.GMMs)
            options.modelPc_g = dlmread([Statdir,'gpm_gmi_global_cov_model_pc_u.txt']);
            
            %GMM = loadobj(sat01_GMModels.mat);
            load gpmgmi_60Xpc_XGMModels.mat GMModels;
        end
        
        if (options.kmeans)
            options.generators = zeros(options.nr,length(options.obs_vars));
            % Simu-BT --> distinct regions
            for i=1:options.nr
                options.generators(i,:) = dlmread(['sat01_rgn',num2str(i,'%0.2d'),'_generator.txt']);
            end
        end
        
    end
    
    % -------------- Declare Useful Statistical Variables -- ---------------    
    % observation principal components, used for computing regions!!!!
    options.obsPc = cell(options.nr,1);
    % model and obs CCVs, used for computing NL operator
    options.modelCcv = cell(options.nr,1);
    options.obsCcv = cell(options.nr,1);
    % the CCV R^2, used for computing the linear operator
    options.ccvS2 = cell(options.nr,1);
    
    % ------------- Calculate the Statistics -------------------------------
    if (options.computeGlobalStats)
        [ model_mean, obs_mean, ncols, options.ndata ] = computeMean_one( options ); %% model_mean(ncol, nregion);obs_mean(nchannel, nregion)
        %%%  ncols: How many specific levels/variables a model variable has

        [ model_stddev, obs_stddev ] = computeVariance_one(options, model_mean, obs_mean);

        [ model_cov, obs_cov, model_obs_ccov ] = computeCovariance_one(options, model_mean, obs_mean, model_stddev, obs_stddev);
        
    elseif (~options.computeGlobalStats) && (options.GMMs)
        [ model_mean, obs_mean, ncols, options.ndata ] = computeMean_one( options, GMModels ); 
       
        [ model_stddev, obs_stddev ] = computeVariance_one(options, model_mean, obs_mean, GMModels);

        [ model_cov, obs_cov, model_obs_ccov ] = computeCovariance_one(options, model_mean, obs_mean, model_stddev, obs_stddev, GMModels);
    end
    % write it to file
    writeStatsToFile_one(options,model_mean,obs_mean,model_stddev,obs_stddev,model_cov,obs_cov,model_obs_ccov);
    
    % array of ones for the number of columns, channels.  Used for plotting.
    nchans = ones(size(obs_mean,1),1);
    
    
    % -------------------now compute the CCVs------------------------------
    for reg=1:options.nr
        close all;        
        
        % plot the model and obs covariance matrix
        if (options.computeGlobalStats)
            rgnstr = '-global';
        else
            rgnstr = ['-xGMM',num2str(reg,'%0.2d')];
        end
        
        %%% Standarized data used
        Cxx = squeeze(model_cov(reg,:,:));
        Cyy = squeeze(obs_cov(reg,:,:));
        Cxy = squeeze(model_obs_ccov(reg,:,:));
        
        doCovarModelPlot(Cxx, Statdir, ncols,  ['model',rgnstr], ' (left/top=surface)',options.model_vars);
        doCovarObsPlot(Cyy, Statdir, nchans, ['obs',rgnstr], '',options.obs_vars);    
      
        % compute the SVD of all three matrices
        [Ux,Sx2] = svds(Cxx,options.nmodelsvd_max);  %% [U,S] = svds(A, k):returns the k largest singular values of A, returns the left singular vectors U, returns singular values as Sx2 %options.nmodelsvd_max = 300
        [Uy,Sy2] = svds(Cyy,options.nobssvd_max); %% options.nobssvd_max = 9;
 
        Sx2d = diag(Sx2); %% construct diagonal matrix with singular value
        Sy2d = diag(Sy2);

        % plot the spread of the SVDs
%       doSVDCumVarPlot(diag(Sx2),['model',rgnstr]);
%       doSVDCumVarPlot(diag(Sy2),['obs',rgnstr]);

        %%% specify tolerance value 1: if singular value>1, corresponding vectors retained
        %%% e.g. Got 50 eigenvectors from SVDS(Cxx), retain some based on threshold.
        nmodelsvd = max(length(Sx2d(Sx2d > 1)),options.nmodelsvd_min); %%options.nmodelsvd_min = 70; C = max(A,B) returns an array with the largest elements taken from A or B
        nobssvd = max(length(Sy2d(Sy2d > 1)),options.nobssvd_min); %%options.nobssvd_min = 9;
        
        disp(['Keeping ',num2str(nmodelsvd),' X pcs.'])
        disp(['Keeping ',num2str(nobssvd),' Y pcs.'])
        %%% Keeping 123 principle components: keep 123 sinular vectors/values

        %%% recude dimensionality
        Ux = Ux(:,1:nmodelsvd);
        Uy = Uy(:,1:nobssvd);
        Sx2 = Sx2(1:nmodelsvd,1:nmodelsvd);
        Sy2 = Sy2(1:nobssvd,1:nobssvd);
        Sx = sqrt(Sx2); 
        Sy = sqrt(Sy2);
        
        if (options.computeGlobalStats)
            filePart = 'global';
        else
            filePart = ['xGMM',num2str(reg,'%0.2d')];
        end
       
        model_pc_u_file = [Statdir,'gpm_gmi_',filePart,'_cov_model_pc_u.txt']; %% pc: principle component
        obs_pc_u_file = [Statdir,'gpm_gmi_',filePart,'_cov_obs_pc_u.txt'];
        model_pc_s_file = [Statdir,'gpm_gmi_',filePart,'_cov_model_pc_s.txt'];
        obs_pc_s_file = [Statdir,'gpm_gmi_',filePart,'_cov_obs_pc_s.txt'];
        dlmwrite(model_pc_u_file,Ux); %% dlmwrite(filename,M) writes numeric data in array M to an ASCII format file
        dlmwrite(obs_pc_u_file,Uy);
        dlmwrite(model_pc_s_file,Sx); 
        dlmwrite(obs_pc_s_file,Sy);
 
        % save obs PC for computing regions
        options.modelPc{reg} = Ux;

        % plot the first 4 principal components of model and obs
        %doPCPlot_one(options.model_vars, Statdir, 'PC', Ux, diag(Sx2), ncols,  min(4,nmodelsvd), ['model',rgnstr], false);
        %doPCPlot_one(options.obs_vars, Statdir, 'PC', Uy, diag(Sy2), nchans, min(4,nobssvd), ['obs-bt',rgnstr], false);

        % ----------------compute the CCVs---------------------------------
        Cxyprime = (Sx\Ux')*Cxy*(Uy/Sy);
        
        Cxyprime_file = [Statdir,'gpm_gmi_',filePart,'_cxyprime.txt'];
        dlmwrite(Cxyprime_file,Cxyprime);
        
        [Uc,Sc2,Vc] = svds(Cxyprime,max(nmodelsvd,nobssvd));
        Sxd = diag(Sx);
        Syd = diag(Sy);
        Sc2d = diag(Sc2);
               
        model_obs_ccov_uc_file = [Statdir,'gpm_gmi_',filePart,'_ccov_model_svd.txt'];
        model_obs_ccov_vc_file = [Statdir,'gpm_gmi_',filePart,'_ccov_obs_svd.txt']; 
        dlmwrite(model_obs_ccov_uc_file,Uc);
        dlmwrite(model_obs_ccov_vc_file,Vc);

        A = Ux*(Sx\Uc);
        B = Uy*(Sy\Vc);
        
        % save the CCVs for computing NL data
        options.modelCcv{reg} = A;
        options.obsCcv{reg} = B;
        options.ccvS2{reg} = Sc2d;

        nccv = min(length(Sc2d),options.maxCcvs); %%% !!!! decide final amount of CCVs
        
        %cc = Sc2d.^2;
        
        %corr_file = [Statdir,'global_CCV','_correlation.txt'];        
        %dlmwrite(corr_file,cc);

        
%         % ---------------------plot the CCVs-------------------------------
        %doPCPlot_one(options.model_vars, Statdir,'CCV', A, Sc2d.^2, ncols,  nccv, ['model',rgnstr],    true);
%         %%% function doPCPlot( vars, titlestr, U, S, ncols, numToPlot, type, isCcv )
        %doPCPlot_one(options.obs_vars, Statdir, 'CCV', B, Sc2d.^2, nchans, nccv, ['obs-bt',rgnstr], true);
        
        % Save CCVs to txt files
        for i=1:nccv
            if (options.computeGlobalStats)
                filePart = 'global';
            else
                filePart = ['xGMM',num2str(reg,'%0.2d')];
            end

            ccvPart = ['_ccv',num2str(i,'%0.2d')];
            
            prefix = [Statdir,'gpm_gmi_',filePart,ccvPart];

            model_ccv_file = [prefix,'_model.txt'];
            obs_ccv_file = [prefix,'_obs.txt'];
            
            ccv_s_file = [Statdir,'gpm_gmi_',filePart,'_ccv',num2str(i,'%0.2d'),'_r.txt'];
            dlmwrite(model_ccv_file,A(:,i),'precision','%15.15f');
            dlmwrite(obs_ccv_file,B(:,i),'precision','%15.15f');
            dlmwrite(ccv_s_file,Sc2d(i),'precision','%15.15f');
            disp(['Saved model/obs ccvs for region ',num2str(reg),' to files ',model_ccv_file, ', ',obs_ccv_file, ' and ', ccv_s_file])        

            scatPoint = 1;

            exps = options.exps_all{scatPoint};
        end
    end    

    
    % find the regions to use
    if (options.computeGlobalStats)
            xpcdata = compute_XPC_Data( options, model_mean, model_stddev, options.GMMsnpc );
            
            % Gaussian Mixture Models
            disp(['Begin GMM clustering based on global model data:'])
            if (options.GMMs)
                goptions = statset('MaxIter',1000,'Display','final','FunValCheck','on');
                rng(1); % For reproducibilit
                GMModels = fitgmdist(xpcdata{1}(1:options.GMMsnpc,:)',options.nregions,'Options',goptions);
                save gpmgmi_60Xpc_XGMModels GMModels 
            end

            % k-means clustering
            if (options.kmeans)
                soptions = statset('MaxIter',200,'Display','iter');
                %%% statset creates statistics options structure

                [~,c] = kmeans(ypcdata{1}(1:3,:)',options.nregions,'replicates',10,...
                     'Options',soptions);
                %%% c[4,3]: centroid location-- rows: clusters; columns: variables
                %%% 'Replicates' — Number of times to repeat clustering using new initial cluster centroid positions

                cPrime = c*options.obsPc{1}(:,1:3)'; 
                %%% cPrime[4,7] 

                for i=1:options.nregions
                    region_file = [Statdir,'sat01_rgn',num2str(i,'%0.2d'),'_generator.txt'];
                    dlmwrite(region_file,cPrime(i,:)','precision','%15.15f');            
                end

                dlmwrite([Statdir,'regions4.txt'],cPrime);
            end 
    end
% 
%    % compute sample correltion of linear combination of AX and BY
%    if (~options.computeGlobalStats)
%        % % sample correlation of training dataset
%        [ xpair, ypair, rgnpro ] = ccvpair_one( options, model_mean, obs_mean, model_stddev, obs_stddev, GMModels);
%        xpair_lc = zeros(nccv,sum(options.ndata));
%        ypair_lc = zeros(nccv,sum(options.ndata));
%        npscatter = 10000;
%        inds = zeros(nccv, npscatter);
%        xsample = zeros(nccv,npscatter); 
%        ysample = zeros(nccv,npscatter); 
%
%            %%% linear combination of A^T X and B^T Y
%        for ncv=1:nccv
%            for k=1:options.numGMM
%                xpair_lc(ncv,:) = xpair_lc(ncv,:)+rgnpro{k}(1,:).*xpair{k}(ncv,:);
%                ypair_lc(ncv,:) = ypair_lc(ncv,:)+rgnpro{k}(1,:).*ypair{k}(ncv,:);
%            end
%
%            inds(ncv,:) = randperm(numel(ypair_lc(ncv,:)),npscatter);
%            ind = inds(ncv,:);
%
%            xsample(ncv,:) = xpair_lc(ncv,ind); 
%            ysample(ncv,:) = ypair_lc(ncv,ind); 
%        end
%
%
%        % % sample correlation of modeled dataset
%        %%% Predict B^T y for all GMM components
%        ypln = cell(options.numGMM,1);
%        ypnln = cell(options.numGMM,1);
%
%        for i=1:options.numGMM 
%            ypln{i} = zeros(nccv,npscatter);  
%            ypnln{i} = zeros(nccv,npscatter); 
%        end
%
%        for k=1:options.numGMM 
%            for ncv=1:nccv
%                ind = inds(ncv,:);
%                xcsample = xpair{k}(ncv,ind);
%                ycsample = ypair{k}(ncv,ind);
%                ypln{k}(ncv,:) = xcsample*options.ccvS2{k}(ncv); % linear
%                ypnln{k}(ncv,:) = nlFunc(xcsample',xcsample',ycsample',10)'; %nonlinear
%            end
%        end
%    
%        %%% linearly combined predicted B^T y
%        ypln_syn = zeros(nccv, npscatter);
%        ypnln_syn = zeros(nccv, npscatter);
%        for ncv=1:nccv
%            for k=1:options.numGMM
%                ind = inds(ncv,:);
%                ypln_syn(ncv,:) = ypln_syn(ncv,:)+rgnpro{k}(1,ind).*ypln{k}(ncv,:);
%                ypnln_syn(ncv,:) = ypnln_syn(ncv,:)+rgnpro{k}(1,ind).*ypnln{k}(ncv,:);
%            end
%        end
%
%        %%% calculate statistics
%        corr_ypln = zeros(1,nccv);
%        corr_ypnln = zeros(1,nccv);
%        for ncv=1:nccv
%            corr_ln = corrcoef(ysample(ncv,:)', ypln_syn(ncv,:)');
%            corr_ypln(1,ncv) =  corr_ln(1,2);
%            corr_nln = corrcoef(ysample(ncv,:)', ypnln_syn(ncv,:)');
%            corr_ypnln(1,ncv) =  corr_nln(1,2);
%        end
%        disp(['correlation of linear operator', num2str(corr_ypln)])
%        disp(['correlation of nonlinear operator', num2str(corr_ypnln)]);
%   
%        for ncv=1:nccv
%            h1 = figure;
%            h1.PaperPositionMode = 'auto';
%
%            pointsize = 30;
%
%            scatter(ysample(ncv,:)', ypln_syn(ncv,:)', pointsize);
%            hold on;
%            scatter(ysample(ncv,:)', ypnln_syn(ncv,:)', pointsize,'red');
%
%
%            minaxis = min(min(ysample(ncv,:)),min(ypln_syn(ncv,:)));
%            minaxis = min(minaxis, min(ypnln_syn(ncv,:)));  
%            maxaxis = max(max(ysample(ncv,:)),max(ypln_syn(ncv,:)));
%            maxaxis = max(maxaxis, max(ypnln_syn(ncv,:)));
%
%            x = linspace(minaxis,maxaxis);
%            y = linspace(minaxis, maxaxis);
%            plot(x,y,'black', 'LineWidth',2);
%
%
%            xLimits = [minaxis maxaxis];
%            yLimits = [minaxis maxaxis];
%            set(gca,'Xlim', xLimits, 'Ylim',yLimits, 'DataAspectRatio', [1 1 1]);
%            set(gca, 'linewidth',1);
%
%
%            %set(gca,'xtick', [minT:25:maxT],'FontSize', 20);
%            %set(gca,'ytick', [minT:25:maxT],'FontSize', 20);
%            xlabel({'Actual B^T Y'''},'FontSize', 16)
%            ylabel({'Regressed B^T Y'''},'FontSize', 16)
%        
%            istr = num2str(ncv);
%            legendln = ['Lin fit, R^2:',num2str(100*corr_ypln(ncv)), '%']
%            legendnln = ['NL fit, R^2:',num2str(100*corr_ypnln(ncv)), '%']
%            %legend({'Lin fit, R^2: ','NL fit'},'Location','SouthEast')
%            legend({legendln, legendnln},'Location','SouthEast')
%            title(['Scatter plot of CCV #',istr],'FontSize',20)
%            set(gca,'FontSize',16)
%            set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
%            ccvPart = ['_ccv',num2str(ncv,'%0.2d')];
%            saveas(gcf, ['BtYcompare_',ccvPart,'.jpg']);
%
%            %set(gcf,'renderer','opengl') %gcf:current figure handle   
%        end
%  
%    end
%% 
%     



 % ----plot the scatter plot of the CCVs back on the original data
% note that this is not a very good validation of the method
% if (options.computeGlobalStats)
%     [xdata, ydata] = computeNLData_one( options, model_mean, obs_mean, model_stddev, obs_stddev );
%    
% elseif (~options.computeGlobalStats) && (options.GMMs)
%     [xdata, ydata] = computeNLData_one( options, model_mean, obs_mean, model_stddev, obs_stddev, GMModels);
%     
% end
%     %%% xdata{i} = zeros(nccv,options.ndata(i))
%     %%% obtain: canonical correlation relationships without dimensionality reduction
%     %%% xdata: AX' (all columns) ydata: BY' (all columns/points)
%     
%     
%     for k=1:options.nr
%         % plot the model and obs covariance matrix
%         if (options.computeGlobalStats)
%             rgnstr = '-global';
%         else
%             rgnstr = ['-rgn',num2str(k,'%0.2d')];
%         end
%          
%         for ccv=1:nccv
%             figure
%             
%             % limit number of scatter points in the graph
%             npscatter = min(10000,length(ydata{k}(ccv,:)));
%             inds = randperm(numel(ydata{k}(ccv,:)),npscatter);
%             %%% numel(A): returns the number of elements in array A
%             %%% randperm(n,k):returns a row vector containing k unique integers selected randomly from 1 to n
%             
%             ysample = ydata{k}(ccv,inds); 
%             xsample = xdata{k}(ccv,inds);
%             
%             % predicted BY' using operator (size: 10000)
%             yplinear = xsample*options.ccvS2{k}(ccv); %%options.ccvS2{k}(ccv): scalar
%             ypnl = nlFunc(xsample',xsample',ysample',10)'; 
%             
%             % predicted BY' for plotting (size: 1000)
%             xnlplot = linspace(min(xdata{k}(ccv,:))-0.5,max(xdata{k}(ccv,:))+0.5,1000);
%             [ynlplot,ynlderplot] = nlFunc(xnlplot',xdata{k}(ccv,:)',ydata{k}(ccv,:)',10);
%             ynlplot = ynlplot';
%             ynlderplot = ynlderplot';
%             ylinplot = xnlplot*options.ccvS2{k}(ccv);
%             
%             
%             % scatter plot x versus y 
%             plot(xnlplot,ylinplot,'-k',xnlplot,ynlplot,'-r','LineWidth',2)
%             hold on
%             cloudPlot(xdata{k}(ccv,:),ydata{k}(ccv,:),[],false,[200,200]); colorbar
%             SSerr = sum((yplinear-ysample).^2);
%             SStot = sum(ysample.^2);
%             r2v = 1 - SSerr/SStot;
%             fprintf('Linear R^2 actual: %f\n',r2v);
%             SSerr = sum((ypnl-ysample).^2);
%             r2v = 1 - SSerr/SStot;
%             fprintf('NL R^2 actual: %f\n',r2v);
%             fprintf('Linear R^2 theory: %f\n',options.ccvS2{k}(ccv)^2);
%             
%             xoutind = abs(xdata{k}(ccv,:)) > 6;
%             xoutliers = xdata{k}(ccv,xoutind);            
%             youtliers = ydata{k}(ccv,xoutind);
%             
%             scatter(xoutliers,youtliers);
% 
%             istr = num2str(ccv);
%             title(['Scatter plot of ',rgnstr(2:end),' CCV #',istr, ' R^2: ',num2str(100*r2v), '%'],'FontSize',20)
%             xlabel(['a_',istr,'^T X'''],'FontSize',16);
%             ylabel(['b_',istr,'^T Y'''],'FontSize',16);
%             set(gca,'FontSize',16)
% 
%             ccvPart = ['_ccv',num2str(ccv,'%0.2d')];
% 
%             set(gcf,'renderer','opengl') %gcf:current figure handle
%             legend({'Lin fit','NL fit','PDF','Outliers'},'Location','SouthEast')
%             saveas(gcf,['nl_scatter_',rgnstr(2:end),ccvPart,'.eps'],'psc2');
%                         
%             prefix = ['sat01_',rgnstr(2:end),ccvPart];
%             
%             dlmwrite([prefix,'_nl_x.txt'],xnlplot','precision','%15.15f')
%             dlmwrite([prefix,'_nl_y.txt'],ynlplot','precision','%15.15f')
%             dlmwrite([prefix,'_nl_y_der.txt'],ynlderplot','precision','%15.15f')
%             
%             %--------------------------hist--------------------------------
%             figure
%             [nelem,xcenters] = hist(ydata{k}(ccv,:),min(40,round(length(ydata{k}(ccv,:))/5)));
%             bar(xcenters,nelem);
% 
%             [~,maxInd] = max(nelem);
%             modeVal = xcenters(maxInd);
%             
%             meanVal = mean(ydata{k}(ccv,:));
%             stdVal = std(ydata{k}(ccv,:));
% 
%             xlabel(['CCV ',istr],'FontSize',16);
%             ylabel('Count','FontSize',16);
%             set(gca,'FontSize',16);
%             if (options.computeGlobalStats)
%                 title(['Global CCV #',istr,' (mean:',num2str(meanVal,'%7.4f'),', std:',num2str(stdVal,'%7.4f'),', mode:',num2str(modeVal,'%7.4f'),')'],'FontSize',20);
%                 saveas(gcf,['hist_train',rgnstr,ccvPart,'.eps'],'psc2');                        
%             else
%                 title(['Region ',rgnstr(2:end),' CCV #',istr,' (mean:',num2str(meanVal,'%7.4f'),', std:',num2str(stdVal,'%7.4f'),', mode:',num2str(modeVal,'%7.4f'),')'],'FontSize',20);
%                 saveas(gcf,['hist_train',rgnstr,ccvPart,'.eps'],'psc2');   
%                          
%             end
%         end
%     end
    

    
