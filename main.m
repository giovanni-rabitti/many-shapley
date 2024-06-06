% Medical Insurance Premium Prediction 
% https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction


%% Neural Network

x= Medicalpremium(:,1:end-1);
y= Medicalpremium(:,end);
k = size(x,2);
n =size(x,1);
L=char({'age', 'diabetes', 'blood pressure problems', 'any transplants', 'any chronic diseases', 'height', 'weight', 'known allergies','cancer in family', 'number of surgeries'});

%y(1)/MedicalNeuralNetworkFunction(x(1,:))
[Shap,Val,ST,more]=finitechanheshapmobMEDICAL(x(1,:),x(2,:))
more.ShapOwen(7,1)
(MedicalNeuralNetworkFunction(x(2,:))- MedicalNeuralNetworkFunction(x(1,:)))/more.ShapOwen(7,1)


%% Baseline Shapley value

baselinemedium=zeros(n,k);
medium=mean(x,1);
baselinemediumshapleyowen=zeros(n,k,k);
for i=1:n
    [Shap,Val,ST,more]=finitechanheshapmobMEDICAL(medium,x(i,:));
    baselinemedium(i,:)=Shap;
    baselinemediumshapleyowen(i,:,:)=more.ShapOwen;
    i
end


figure
plot(x(:,1), baselinemedium(:,1:5),'o','LineWidth',1.2)
hold on
plot(x(:,1), baselinemedium(:,6:10),'*')
legend({'age','diabetes', 'blood pressure problems', 'any transplants', 'any chronic diseases', 'height', 'weight', 'known allergies','cancer in family', 'number of surgeries'})
xlabel('age')
ylabel('Sh_i')



% finite change Shap owen interaction
mean(baselinemediumshapleyowen,1);
SHOW=zeros(k,k);
for i=1:k
    for j=1:k
        SHOW(i,j)=mean(baselinemediumshapleyowen(:,i,j));
    end
end
    

figure
plot(x(:,1), baselinemediumshapleyowen(:,8,1),'.','MarkerSize',12)
hold on
plot(x(:,1), baselinemediumshapleyowen(:,3,1),'.','MarkerSize',12)
hold on
plot(x(:,1), baselinemediumshapleyowen(:,10,1), '.','MarkerSize',12)
hold on
plot(x(:,1), baselinemediumshapleyowen(:,10,8), '.','MarkerSize',12)
hold on
plot(x(:,1), baselinemediumshapleyowen(:,5,1), '.','MarkerSize',12)
hold on
plot(x(:,1), baselinemediumshapleyowen(:,8,3), '.','MarkerSize',12)
xlabel('age')
ylabel('Sh_{i,j}')
title('Finite-change Shapley-Owen values')
legend({'age-allergies','age-blood pressure', 'age-surgery', 'allergies-surgery', 'age-chronic disease', 'allergies-blood problems' })
%%  difference between min and max
[valmin, posmin]=min(y);
[valmax, posmax]=max(y);
 [Shap,Val,ST,more]=finitechanheshapmobMEDICAL(x(posmin,:),x(posmax,:));


 %% Finite change Shaple value between min and max, for all fixed ages

 annitotali=66-17;
 res=zeros(annitotali,k);
 for j=1:annitotali%j=18:66
     jj=j+17;
     index=find((x(:,1)==jj));
     [valmin1, posmin1]=min(y(index));
     [valmax1, posmax1]=max(y(index));
   [Shap,Val,ST,more]=finitechanheshapmobMEDICAL(x(index(posmin1),:),x(index(posmax1),:));
    res(j,:)=Shap;
 end
 
 figure
 %plot(18:66, res(:,2:end),'.','MarkerSize',20)
plot( 18:66, res(:,2:4),'.','MarkerSize',20)
hold on
plot(18:66, res(:,5:7),'o','LineWidth',1.5)
hold on
plot(18:66, res(:,8:10),'+')
 legend({'diabetes', 'blood pressure problems', 'any transplants', 'any chronic diseases', 'height', 'weight', 'known allergies','cancer in family', 'number of surgeries'})
 xlabel('age')
 ylabel('Sh_i')
 
 %% doppio grafico
 figure
 subplot(2,1,1)
plot(x(:,1), baselinemedium(:,1:5),'o','LineWidth',1.2)
hold on
plot(x(:,1), baselinemedium(:,6:10),'*')
legend({'age','diabetes', 'blood pressure problems', 'any transplants', 'any chronic diseases', 'height', 'weight', 'known allergies','cancer in family', 'number of surgeries'})
xlabel('age')
ylabel('Sh_i')
title('Finite-change Shapley values')


Shapsquared=zeros(n,k);
ShapOwensquared=zeros(n,k,k);

%save('Shapsquared.dat','-mat','Shapsquared');
%save('ShapOwensquared.dat','-mat','ShapOwensquared')
%load('Shapsquared.dat','-mat')
%load('ShapOwensquared.dat','-mat')

%m=mean(MedicalNeuralNetworkFunction(x(1:n,:)));

tic
for i=1:n
   % [Shapsquared(i,:),Val,ST,more]=squaredcohortshapmobMEDICAL(x,x(i,:),m);
   [Shapsquared(i,:),Val,ST,more]=squaredcohortshapmobMEDICALcohorts(y,x,x(i,:),d); 
   ShapOwensquared(i,:,:)=more.ShapOwen;
    i
end 
toc
ShapEffects=mean(Shapsquared,1);


subplot(2,1,2)
plot(x(:,1), Shapsquared(:,1:5),'o','LineWidth',1.2)
hold on
plot(x(:,1), Shapsquared(:,6:10),'*')
legend({'age','diabetes', 'blood pressure problems', 'any transplants', 'any chronic diseases', 'height', 'weight', 'known allergies','cancer in family', 'number of surgeries'})
xlabel('age')
ylabel('Sh_i')
title('Squared cohort Shapley values')
 
 %% Squared cohort shapley value

d=0.1.*range(x);
Shapsquared=zeros(n,k);
ShapOwensquared=zeros(n,k,k);

%save('Shapsquared.dat','-mat','Shapsquared');
%save('ShapOwensquared.dat','-mat','ShapOwensquared')
%load('Shapsquared.dat','-mat')
%load('ShapOwensquared.dat','-mat')

%m=mean(MedicalNeuralNetworkFunction(x(1:n,:)));

tic
for i=1:n
   % [Shapsquared(i,:),Val,ST,more]=squaredcohortshapmobMEDICAL(x,x(i,:),m);
   [Shapsquared(i,:),Val,ST,more]=squaredcohortshapmobMEDICALcohorts(y,x,x(i,:),d); 
   ShapOwensquared(i,:,:)=more.ShapOwen;
    i
end 
toc
ShapEffects=mean(Shapsquared,1);

ShapOwen=zeros(10,10);
 for i=1:10
     for j=1:10
         ShapOwen(i,j)=mean(ShapOwensquared(:,i,j));
     end
 end 

figure
plot(x(:,1),Shapsquared,'.','MarkerSize',12)
legend({'age','diabetes', 'blood pressure problems', 'any transplants', 'any chronic diseases', 'height', 'weight', 'known allergies','cancer in family', 'number of surgeries'})

figure
plot(x(:,1), Shapsquared(:,1:5),'o','LineWidth',1.2)
hold on
plot(x(:,1), Shapsquared(:,6:10),'*')
legend({'age','diabetes', 'blood pressure problems', 'any transplants', 'any chronic diseases', 'height', 'weight', 'known allergies','cancer in family', 'number of surgeries'})
xlabel('age')
ylabel('Sh_i')

figure
id=x(:,1)<=29;
%plot(x(id,1),Shapsquared(x(:,1)<=29),'.','MarkerSize',12)
bar(mean(Shapsquared(id,:),1))

% shapley effecte per i 30-40 enni
figure
id=x(:,1)>=30 & x(:,1)<41;
%plot(x(id,1),Shapsquared(x(:,1)<=29),'.','MarkerSize',12)
bar(mean(Shapsquared(id,:),1)) 
set(gca, 'XTick', 1:k); 
set(gca, 'XTickLabel', L);
set(gca,'XTickLabelRotation',45)
title('Shapley effects for 30-40 yo')

figure
bar(ShapEffects)
%set(gca,'XTickLabelRotation',90)
set(gca, 'XTick', 1:k); 
set(gca, 'XTickLabel', L);
set(gca,'XTickLabelRotation',45)
title('Shapley effects')


%plot interactions of age
figure
bar(ShapOwen(2:end,1))
set(gca, 'XTick', 1:k-1); 
Lage=char({'diabetes', 'blood pressure problems', 'any transplants', 'any chronic diseases', 'height', 'weight', 'known allergies','cancer in family', 'number of surgeries'});
set(gca, 'XTickLabel', Lage);
set(gca,'XTickLabelRotation',45)
ylabel('Sh_{i,j}')
title('Shapley-Owen effects for age')

figure
plot(x(:,1),ShapOwensquared(:,10,1),'.', 'MarkerSize', 15)
idnop=x(:,10)==0; idop1=x(:,10)==1;
idop2=x(:,10)==2; idop3=x(:,10)==3;

% plot interazione age e number of surgeries 
figure
plot(x(idnop,1),ShapOwensquared(idnop,10,1),'.', 'MarkerSize', 15)
hold on
plot(x(idop1,1),ShapOwensquared(idop1,10,1),'.', 'MarkerSize', 15)
hold on
plot(x(idop2,1),ShapOwensquared(idop2,10,1),'.', 'MarkerSize', 15)
hold on
plot(x(idop3,1),ShapOwensquared(idop3,10,1),'.', 'MarkerSize', 15)
legend('no surgery', '1 surgery', '2 surgeries', '3 surgeries')
xlabel('age')
ylabel('Sh_{i,j}')
title('Squared-cohort Shapely-Owen for age and number of surgeries')


% shapley owen matrix
matr = ShapOwen+ShapOwen';
for i=1:k
    matr(i,i)=mean(Shapsquared(:,i));
end


 %% Interaction between age and weight
 
xage= sortrows(x,1); %sort by the first column 
medium=mean(x); 
testage=zeros(n);
for i=1:n
    [Shap,Val,ST,more]=finitechanheshapmobMEDICAL(medium,xage(i,:));
    testint(i)=more.ShapOwen(7,1);
    i
end
figure
plot(abs(testint)) % dopo i xage(700,1)=51 anni le interazioni tre age e weight sono più grandi

plot(x(:,1),testint,'o')

%
testint=zeros(n-1);
for i=1:n-1
    [Shap,Val,ST,more]=finitechanheshapmobMEDICAL(xage(i,:),xage(i+1,:));
    testint(i)=more.ShapOwen(5,2)/sign(Val);
    i
end
figure
plot(testint)

%% comparison cohort and neural network


ShapsquaredNN=zeros(n,k);
ShapOwensquaredNN=zeros(n,k,k);

m=mean(MedicalNeuralNetworkFunction(x(1:n,:)));

tic
for i=1:n
    [ShapsquaredNN(i,:),Val,ST,more]=squaredcohortshapmobMEDICAL(x,x(i,:),m);
   %[Shapsquared(i,:),Val,ST,more]=squaredcohortshapmobMEDICALcohorts(y,x,x(i,:),d); 
   ShapOwensquaredNN(i,:,:)=more.ShapOwen;
    i
end 
toc


 figure
 subplot(2,1,1)
plot(x(:,1), Shapsquared(:,1:5),'o','LineWidth',1.2)
hold on
plot(x(:,1), Shapsquared(:,6:10),'*')
%legend({'age','diabetes', 'blood pressure problems', 'any transplants', 'any chronic diseases', 'height', 'weight', 'known allergies','cancer in family', 'number of surgeries'})
xlabel('age')
ylabel('Sh_i')
title('Squared cohort Shapley values from given data')

subplot(2,1,2)
plot(x(:,1), ShapsquaredNN(:,1:5),'o','LineWidth',1.2)
hold on
plot(x(:,1), ShapsquaredNN(:,6:10),'*')
legend({'age','diabetes', 'blood pressure problems', 'any transplants', 'any chronic diseases', 'height', 'weight', 'known allergies','cancer in family', 'number of surgeries'})
xlabel('age')
ylabel('Sh_i')
title('Squared cohort Shapley values from NN')

%%
annitotali=66-17;
 res=zeros(annitotali,k);
 int=zeros(annitotali,10,10);
 for j=1:annitotali%j=18:66
     jj=j+17;
     index=find((x(:,1)==jj));
     [valmin1, posmin1]=min(y(index));
     [valmax1, posmax1]=max(y(index));
   [Shap,Val,ST,more]=finitechanheshapmobMEDICAL(x(index(posmin1),:),x(index(posmax1),:));
    res(j,:)=Shap;
    int(j,:,:)=more.ShapOwen+more.ShapOwen';
 end
 
 medieint=zeros(10,10);
 for i=1:10
     for j=1:10
         medieint(i,j)=mean(int(:,i,j));
     end
 end 


 