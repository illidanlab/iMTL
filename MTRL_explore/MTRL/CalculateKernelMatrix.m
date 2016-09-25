function kernelmatrix=CalculateKernelMatrix(varargin)
    if nargin==3
        data=varargin{1};
        kerneltype=varargin{2};
        kernelpar=varargin{3};
        n=size(data,1);
        if strcmp(kerneltype,'linear')==1
            kernelmatrix=data*data';
        elseif strcmp(kerneltype,'poly')==1
            kernelmatrix=(data*data').^kernelpar;
        else
            kernelmatrix=zeros(n);
            for i=1:n
                for j=i:n
                    kernelmatrix(i,j)=KernelFunction(data(i,:),data(j,:),kerneltype,kernelpar);
                    kernelmatrix(j,i)=kernelmatrix(i,j);
                end
            end
        end
        clear data;
    elseif nargin==4
        data1=varargin{1};
        data2=varargin{2};
        kerneltype=varargin{3};
        kernelpar=varargin{4};
        if strcmp(kerneltype,'linear')==1
            kernelmatrix=data1*data2';
        elseif strcmp(kerneltype,'poly')==1
            kernelmatrix=(data1*data2').^kernelpar;
        else
            m=size(data1,1);
            n=size(data2,1);
            kernelmatrix=zeros(m,n);
            for i=1:m
                for j=1:n
                    kernelmatrix(i,j)=KernelFunction(data1(i,:),data2(j,:),kerneltype,kernelpar);
                end
            end
        end
        clear data1 data2;
    end
    clear varargin;
    
function r=KernelFunction(x,y,kernel,para)
    if strcmp(kernel,'rbf')==1
        tmp=x-y;
        r=exp(-(tmp*tmp')/(2*para^2));
    elseif strcmp(kernel,'poly')==1
        r=(x*y').^para;
    elseif strcmp(kernel,'linear')==1
        r=x*y';
    else
        error('The invalid kernel parameters');
    end